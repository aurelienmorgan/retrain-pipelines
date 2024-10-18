![logo_large](https://github.com/user-attachments/assets/19725866-13f9-48c1-b958-35c2e014351a)

# Frameworks
<b>retrain-pipelines</b> is designed to support a variety of modalities and scenarios, with several frameworks being adopted.

To help teams iterate faster, we thought of offering assistance with personal sandbox environment creation. The intent naturally being to draft retraining pipelines fast.

Currently, [Metaflow](https://metaflow.org/) is the first of these frameworks to be integrated.

---

<img src="https://github.com/user-attachments/assets/ecc20501-869d-4159-b5a0-eb0a117520e5" alt="Metaflow" width="200" style="float: right;">

## Metaflow sandbox environments

 However true that <b>Metaflow</b> itself is Apache 2 and free to use, it is cumbersome to install without relying on AWS S3 storage. <u>We fixed that&nbsp;!</u> Below you will find easy-to-follow steps to install <b>Metaflow service+ui</b> for free usage with two possible setups described here.

### fully local<br />
<em><small>(Note that Metaflow isn't supported on native Windows but, all is fine if you go through WSL.)</small></em>

To run <b>Metaflow</b> locally, one needs to first clone both the [metaflow-service](https://github.com/Netflix/metaflow-service)
and [metaflow-ui](https://github.com/Netflix/metaflow-ui) official repos.

We tested with <code>metaflow-service v2.4.0</code> and <code>metaflow-ui v1.3.12</code> but, it's likely that newer versions work for you as well.
<ul>
   <li>
      Create the local artifacts-store directory with adequate permissions via the below CLI command&nbsp;:<br />
      <code>umask 0022 | mkdir -p ${HOME}/local_datastore</code>
   </li>
   <li>
      Launch <em><code>metaflow-service</code></em>&nbsp;:
      <ul>
         <li>
            from within the cloned repo's root folder, adapt the <code>docker-compose.development.yml</code> file by <u>inserting</u> 3 rows as highlighted below&nbsp;:<br />
            <center><img src="https://github.com/user-attachments/assets/853f7ff5-5d43-41c7-ab41-5da3c625ca40" alt="docker-compose development" width=500px /></center>
         </li>
         <li>
            Now, start the service via the below CLI command on that directory&nbsp;:<br />
            <code>docker-compose -f docker-compose.development.yml up</code><br />
            This spins up the <code>PostgreSQL database</code> plus <code>metadata</code> and <code>ui-backend</code> services.
         </li>
      </ul>
   </li>

   <li>
      Launch <em><code>metaflow-ui</code></em>&nbsp;:<br />
      From that other cloned repo's root directory,
      <ul>
         <li>
            Start by building the docker image<br />
            <code>docker build --tag metaflow-ui:latest .</code><br />
            <code>docker image prune -f</code>
         </li>
         <li>
            Launch the container so it uses your <em><code>ui-backend</code></em> service&nbsp;<br />
            <code>docker run -p 3000:3000 -e METAFLOW_SERVICE=http://localhost:8083/ \</code><br />
            <code> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; --name metaflow-ui --rm metaflow-ui:latest</code>
         </li>
      </ul>
   </li>
   <li>
      Browse your fully-local Metaflow UI at
      <a href="http://localhost:3000/" target="_blank">http://localhost:3000/</a>
   </li>
   <li>
      Python API<br />
      Launch the execution of flow runs programatically. 3 different ways to do just that&nbsp;:
        <ul>
           <li>
              raw command with proper set of variables&nbsp;&rArr;<br />
              <code>!export METAFLOW_SERVICE_URL=http://localhost:8080/ && \</code><br />
              <code>&nbsp;export METAFLOW_DEFAULT_METADATA=service && \</code><br />
              <code>&nbsp;export METAFLOW_DEFAULT_DATASTORE='local' && \</code><br />
              <code>&nbsp;export METAFLOW_DATASTORE_SYSROOT_LOCAL={HOME}/local_datastore/ && \</code><br />
              <code>&nbsp;cd {HOME}/local_datastore/ && \</code><br />
              <code>&nbsp;python &lt;fullpath_to_your_flow.py&gt; run</code>
           </li>
           <li>
              <b>retrain-pipelines</b> CLI utility&nbsp;&rArr;<br />
              <code>retrain_pipelines_local &lt;relpath_to_your_flow.py&gt; run</code>
           </li>
           <li>
              <b>retrain-pipelines</b> Jupyter magic&nbsp;&rArr;<br />
              <code>%load_ext retrain_pipelines.local_launcher_magic</code><br />
              <code>%retrain_pipelines_local &lt;relpath_to_your_flow.py&gt; run</code>
           </li>
        </ul>
   </li>
   <li>
      Python SDK
        <ul>
           <li>
              raw import preempted with proper set of variables&nbsp;&rArr;<br />
              <code>import os</code><br />
              <code>os.environ['METAFLOW_SERVICE_URL'] = 'http://localhost:8080/'</code><br />
              <code>os.environ['METAFLOW_DEFAULT_METADATA'] = 'service'</code><br />
              <code>import metaflow</code>
           </li>
           <li>
              <b>retrain-pipelines</b> dedicated python module&nbsp;&rArr;<br />
              <code>from retrain_pipelines.frameworks import local_metaflow as metaflow</code>
           </li>
        </ul>
   </li>
</ul>

<br />

### hosting on <img src="https://github.com/user-attachments/assets/bd56a1bc-9bb7-4699-86fe-e26648d5f62a" width=30 />Google Colab

If you want to develop retraining pipelines for deep learning models and are GPU poor, we've thought of you too&nbsp;!<br />
You can install <b>Metaflow</b> on a CPU Google Colab notebook and use it from another instance of Google Colab notebook which has a GPU runtime.

 - Following the installation/startup instructions as put together in the [metaflow_service.ipynb](./Metaflow/metaflow_service.ipynb). Just press "open in colab" and execute the couple cells in there to spin <b>Statefull Metaflow Service+UI</b> up on Google Colab as well as expose it externally.
 - To get started in developping a retraining pipeline on a separate Google Colab notebook, use [remote_local_metaflow.ipynb](./Metaflow/remote_local_metaflow.ipynb) as a starting point.<br />
There, you'll see how to connect the two. That way, you can run dev pipeline runs using the Google Colab free-tier GPUs !<br />
<em><small>(Note that this option is not advised for high CPU workloads, since clearly this is not the strong suit of Google Colab.)</small></em><br />
For details on the inner-workings of this <b>Statefull Metaflow Service+UI</b> installation, go read <a href="https://huggingface.co/blog/Aurelien-Morgan/stateful-metaflow-on-colab/" target="_blank">our dedicated article on <img src="https://github.com/user-attachments/assets/3ee54f64-0796-4573-9d01-daf26a57c748" width=20/>HuggingFace</a>.

Note that, if unspecified while executing the `startup.sh` script, the default config for this setup is `/data` as MF_ROOT (to host <code>logs</code> and <code>persistent db</code> and <code>artifacts store</code>) and the exposed port number is `7860`. Just remember that this is a setup for which the host for <b>Metaflow</b> and the host for the runtime executing flow runs <u>shall share read/write access to the same MF_ROOT dir (<em>common storage, same fullpath</em>)</u>.

---

