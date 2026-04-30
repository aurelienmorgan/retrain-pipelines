
import gc
import os
import sys
import time


def nuke_dask_cpp():
    """
    In case the dag has already executed on that same python interpreter,
    need to do a hard reset.

    Total Dask + LightGBM C++ cleanup:
    - Close all live dask.distributed clients/scheduler/workers
      (while modules are still importable, so threads shut down cleanly)
    - Remove all Dask/Dask.distributed modules from sys.modules
    - Remove LightGBM modules from sys.modules
    - Delete all globals referencing Dask/LightGBM
    - Force garbage collection
    - Attempt to free C++ resources by clearing LightGBM booster objects
    """

    # 0) Close all live dask.distributed clients BEFORE touching sys.modules.
    #    The dask.distributed.Client created inside dask_regressor_fit (called
    #    from the inline train_model task, i.e. in the main process) is never
    #    explicitly closed, leaving its scheduler thread and worker threads
    #    running against the old module objects.  Nuking sys.modules while
    #    those threads are alive produces two conflicting _BackendData class
    #    identities, which breaks normalize_token on the second execute.
    #    Closing here — while the modules are still importable — lets the
    #    threads reach a clean exit before we remove anything.
    try:
        import distributed

        # Primary path: iterate every registered global client.
        # distributed.client._global_clients is a WeakValueDictionary
        # keyed by client id; copy it to avoid mutation during iteration.
        _gc = getattr(distributed.client, "_global_clients", {})
        for _client in list(_gc.values()):
            try:
                _client.close(timeout=10)
            except Exception:
                pass

        # Belt-and-suspenders: also reach the default client directly.
        try:
            _client = distributed.get_client()
            _client.close(timeout=10)
        except Exception:
            pass

        # Belt-and-suspenders: try the older Client._default singleton path.
        try:
            _client = distributed.client.Client.current()
            _client.close(timeout=10)
        except Exception:
            pass

        # Clear the registry so no stale weak-refs linger.
        try:
            _gc.clear()
        except Exception:
            pass

    except Exception:
        pass

    # Give scheduler / worker threads a moment to reach a clean exit
    # before we remove the modules they may still be referencing.
    time.sleep(0.5)

    # 1) Remove Dask modules from sys.modules.
    #    Capture the live module objects FIRST (keyed by their sys.modules
    #    name) so that step 9 can build an old→new mapping and patch every
    #    module dict that still holds references to them.
    _old_dask_modules = {}
    for mod_name in list(sys.modules):
        if mod_name.startswith("dask") or mod_name.startswith("distributed"):
            _old_dask_modules[mod_name] = sys.modules[mod_name]
            try:
                del sys.modules[mod_name]
            except Exception:
                pass

    # 2) Remove LightGBM modules from sys.modules.
    #    Capture them too: dask_trainer imports `import lightgbm as lgb` at
    #    module level, so dask_trainer.__dict__['lgb'] points to the old
    #    lightgbm module.  lgb.DaskLGBMRegressor.fit() calls into old
    #    lightgbm.dask, which holds `from distributed import default_client`
    #    captured from the OLD distributed whose _global_clients was cleared
    #    in step 0.  The new dask.distributed.Client() created in
    #    dask_regressor_fit registers in the NEW distributed._global_clients,
    #    so old default_client() finds nothing → "No clients found".
    #    Capturing here lets step 9 patch dask_trainer.lgb → fresh lightgbm,
    #    so DaskLGBMRegressor.fit() goes through fresh lightgbm.dask which
    #    uses the same fresh distributed that the Client was created with.
    #    We also evict any third-party module whose name CONTAINS "lightgbm"
    #    (e.g. wandb.integration.lightgbm) so it re-imports fresh and picks
    #    up the new Booster class — step 9 alone cannot fix class-level
    #    references captured at module load time in those third-party modules.
    _old_lgb_modules = {}
    for mod_name in list(sys.modules):
        if mod_name.startswith("lightgbm"):
            _old_lgb_modules[mod_name] = sys.modules[mod_name]
            try:
                del sys.modules[mod_name]
            except Exception:
                pass

    # 3) Delete any globals referencing Dask/LightGBM.
    #    Exclude this function itself: its name contains "dask", so without
    #    the guard it would delete nuke_dask_cpp from globals() on the first
    #    call, making subsequent calls raise NameError.
    #    We retrieve the function's own name from the current frame's code
    #    object — no hard-coding required.
    import inspect as _inspect
    _self_name = _inspect.currentframe().f_code.co_name
    for name in list(globals()):
        if name == _self_name:
            continue
        if "dask" in name.lower() or "lgb" in name.lower() or "lightgbm" in name.lower():
            try: del globals()[name]
            except Exception: pass

    # 4) Force garbage collection
    gc.collect()
    gc.collect()

    # 5) Clear any lingering shared memory files (Linux)
    try:
        shm_dir = "/dev/shm/"
        if os.path.exists(shm_dir):
            for f in os.listdir(shm_dir):
                if "dask" in f.lower() or "lightgbm" in f.lower():
                    try:
                        os.unlink(os.path.join(shm_dir, f))
                    except Exception:
                        pass
    except Exception:
        pass

    # 6) Garbage collection sweep before re-imports
    gc.collect()
    gc.collect()

    # 7) Eagerly re-import core dask modules so that all normalize_token
    #    registrations (including _BackendData) are fully in place in the
    #    main process before any subsequent fork().
    #    Without this, the lazy re-import triggered inside a forked child
    #    may inherit an incomplete or inconsistent dispatch table from the
    #    parent, causing TokenizationError on dd.from_pandas().
    #    dask.distributed MUST be imported here to establish the single
    #    canonical distributed module object in sys.modules BEFORE lightgbm
    #    is re-imported in step 8 below.  If lightgbm were imported first it
    #    would pull in distributed on its own, and then the dask.distributed
    #    import here would be a no-op — which is fine — but the reverse order
    #    guarantees the canonical distributed is the fully-initialised one
    #    from the dask ecosystem, not a side-effect of lightgbm's import path.
    try:
        import dask                   # re-creates normalize_token Dispatch
        import dask.base              # explicit: normalize_token lives here
        import dask.array             # dask_trainer imports dask.array as da
        import dask.dataframe         # re-registers dataframe collections
        import dask_expr._util        # registers _BackendData with normalize_token
        import dask_expr._collection  # registers remaining collection types
        import dask.distributed       # re-initialises distributed machinery;
                                      # must precede lightgbm re-import so that
                                      # lightgbm.dask's `from distributed import
                                      # default_client` binds to THIS object
    except Exception:
        pass

    # 8) Re-import LightGBM AFTER dask/distributed are fully in place.
    #    lightgbm.dask does `from distributed import default_client` at import
    #    time.  By importing lightgbm here — after step 7 has populated
    #    sys.modules['distributed'] with the canonical fresh object — we
    #    guarantee that lightgbm.dask.default_client is bound to the same
    #    distributed.client module that dask.distributed.Client will register
    #    with.  If lightgbm were imported before step 7 (as was the case when
    #    this step was step 5), it would pull in its own fresh distributed
    #    first, and step 7's `import dask.distributed` would be a no-op
    #    pointing at that same object — seemingly fine, but the ordering
    #    dependency means any future refactor could silently re-introduce the
    #    split-brain.  Explicit ordering here makes the invariant robust.
    try:
        import lightgbm as lgb
        # Attempt to free LightGBM C++ state directly.
        # If any LightGBM Booster objects exist, free their memory.
        for obj in gc.get_objects():
            if isinstance(obj, lgb.basic.Booster):
                try:
                    obj.free_network()  # free underlying C++ memory
                    del obj
                except Exception:
                    pass
    except Exception:
        pass

    # 9) Patch stale dask AND lightgbm module references in-place across
    #    every live module dict.
    #
    #    Evicting a module from sys.modules (e.g. dask_trainer) is not
    #    enough: any function object (e.g. dask_regressor_fit) whose
    #    __globals__ IS that module's dict will keep using the old dict even
    #    after the module is evicted.
    #
    #    dask_trainer has module-level imports:
    #        import dask               → __dict__['dask']  = old dask module
    #        import dask.array as da   → __dict__['da']    = old dask.array
    #        import dask.dataframe as dd → __dict__['dd']  = old dask.dataframe
    #        import lightgbm as lgb    → __dict__['lgb']   = old lightgbm
    #
    #    After steps 7+8, sys.modules holds fresh replacements for all of
    #    these.  We build old→new mappings for both dask and lightgbm module
    #    objects and walk every live module's vars(), replacing any attr that
    #    still points to an old object with its fresh counterpart.  Because
    #    __globals__ is the actual module dict (not a copy), this transparently
    #    fixes all live function objects — including cloudpickle-serialized
    #    ones reconstructed in subprocesses — without needing to evict or
    #    reload anything.
    try:
        # Build id(old) → fresh mapping for dask modules.
        _old_to_new = {}
        for _mod_name, _old_mod in _old_dask_modules.items():
            _fresh = sys.modules.get(_mod_name)
            if _fresh is not None and _fresh is not _old_mod:
                _old_to_new[id(_old_mod)] = _fresh

        # Extend with lightgbm modules.
        for _mod_name, _old_mod in _old_lgb_modules.items():
            _fresh = sys.modules.get(_mod_name)
            if _fresh is not None and _fresh is not _old_mod:
                _old_to_new[id(_old_mod)] = _fresh

        # Also map old lightgbm CLASS objects to their fresh counterparts.
        #    Third-party modules such as wandb.integration.lightgbm capture
        #    lightgbm classes (e.g. lightgbm.basic.Booster) as module-level
        #    attributes at import time:
        #        from lightgbm.basic import Booster
        #    or via an isinstance guard:
        #        isinstance(model, lightgbm.basic.Booster)
        #    These are class references, not module references, so the
        #    id(old_module) → fresh_module mapping above does not reach them.
        #    We enumerate every attribute of every captured old lightgbm
        #    module, and for each class/function found there, look up the
        #    same-named attribute in the corresponding fresh module.  If they
        #    differ by identity (old vs new), we add the old→new class mapping
        #    so that step 9's attr-patching loop below will fix any module
        #    that holds the old class directly as an attribute.
        for _mod_name, _old_mod in _old_lgb_modules.items():
            _fresh_mod = sys.modules.get(_mod_name)
            if _fresh_mod is None or _fresh_mod is _old_mod:
                continue
            try:
                for _attr_name, _old_attr in vars(_old_mod).items():
                    if isinstance(_old_attr, type):
                        _fresh_attr = getattr(_fresh_mod, _attr_name, None)
                        if _fresh_attr is not None and \
                           _fresh_attr is not _old_attr:
                            _old_to_new[id(_old_attr)] = _fresh_attr
            except Exception:
                pass

        if _old_to_new:
            for _mod_name in list(sys.modules):
                _mod = sys.modules.get(_mod_name)
                if _mod is None:
                    continue
                # Skip the freshly-imported dask and lightgbm modules
                # themselves — they already have the correct references.
                if _mod_name.startswith("dask") or \
                   _mod_name.startswith("distributed") or \
                   _mod_name.startswith("lightgbm"):
                    continue
                try:
                    for _attr_name, _attr_val in list(vars(_mod).items()):
                        _fresh = _old_to_new.get(id(_attr_val))
                        if _fresh is not None:
                            try:
                                setattr(_mod, _attr_name, _fresh)
                            except Exception:
                                pass
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        # Release the old module objects so they can be GC-ed.
        try:
            del _old_dask_modules
        except Exception:
            pass
        try:
            del _old_lgb_modules
        except Exception:
            pass

    # 🔟 Final garbage collection sweep
    gc.collect()
    gc.collect()

