![logo_large](https://github.com/user-attachments/assets/19725866-13f9-48c1-b958-35c2e014351a)


# Frameworks
<b>retrain-pipelines</b> is designed to support a variety of modalities and scenarios, with several frameworks being adopted.

To help teams iterate faster, we thought of offering assistance with personal sandbox environment creation. The intend naturally being to develop draft retraining pipelines fast.

Currently, [Metaflow](https://metaflow.org/) is the first of these frameworks to be integrated.

---

<img src="https://github.com/user-attachments/assets/24c90e1c-3ae7-4d3c-aaf0-b552bb08266d" alt="Metaflow" width="250" style="float: right;">

## Metaflow sandbox environments

 However true that <b>Metaflow</b> itself is Apache 2 and free to use, it is cumbersome to install without relying on AWS S3 storage. <u>We fixed that&nbsp;!</u> Below you will find easy-to-follow steps to install <b>Metaflow service+ui</b> for free usage.<br />
<em><small>(Note that Metaflow isn't supported on native Windows but, all is fine if you go through WSL.)</small></em>

### fully local
To run <b>Metaflow</b> locally, one needs to first clone both the [metaflow-service](https://github.com/Netflix/metaflow-service)
and [metaflow-ui](https://github.com/Netflix/metaflow-ui) official repos.

We tested with <code>metaflow-service v2.4.0</code> and <code>metaflow-ui v1.3.12</code> but, it's likely that newer versions work for you as well.
<ul>
   <li>
      Create the local artifacts-store directory with adequat permissions via the below CLI command&nbsp;:<br />
      <code>umask 0022 | mkdir -p ${HOME}/local_datastore</code>
   </li>
   <li>
      Launch <em><code>metaflow-service</code></em>&nbsp;:
      <ul>
         <li>
            from within the cloned repo's root folder, adapt the <code>docker-compose.development.yml</code> file by <u>inserting</u> 3 rows as highlighted below&nbsp;:
            <center><img src="https://github.com/user-attachments/assets/9465b1dc-7298-4f9f-8234-8417fa3c3a49" alt="docker-compose development"></center>
         </li>
         <li>
            Now, start the service via the below CLI command on that directory&nbsp;:<br />
            <code>
               docker-compose -f docker-compose.development.yml up
            </code><br />
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
            <code>
               docker build --tag metaflow-ui:latest .<br />
               &nbsp; docker image prune -f
            </code>
         </li>
         <li>
            Launch the container so it uses your <em><code>ui-backend</code></em> service&nbsp;<br />
            <code>
               docker run -p 3000:3000 -e METAFLOW_SERVICE=http://localhost:8083/ --name metaflow-ui --rm metaflow-ui:latest
            </code>
         </li>
      </ul>
   </li>
   <li>
      Browse your fully-local Metaflow UI at
      <a href="http://localhost:3000/" target="_blank">http://localhost:3000/</a>
   </li>
   <li>
      Python SDK
        <ul>
           <li>
              raw<br />
              <code>
                 blabla
              </code>
           </li>
           <li>
              <b>retrain-pipelines</b> CLI utility<br />
              <em>more on this later</em>
           </li>
           <li>
              <b>retrain-pipelines</b> Jupyter magic<br />
              <em>more on this later</em>
           </li>
        </ul>
   </li>
   <li>
      Python API
        <ul>
           <li>
              raw<br />
              <code>
                 blabla
              </code>
           </li>
           <li>
              <b>retrain-pipelines</b> CLI utility<br />
              <em>more on this later</em>
           </li>
           <li>
              <b>retrain-pipelines</b> Jupyter magic<br />
              <em>more on this later</em>
           </li>
        </ul>
   </li>
</ul>

<br />

### hosting on Google Colab

If you want to develop retraining pipelines for deep learning models and are GPU poor, we've thought of you too&nbsp;!<br />
You can install <b>Metaflow</b> on a CPU Google Colab notebook and use it from another instance of Google Colab notebook which has a GPU runtime.

 - Following the installation/startup instructions as detailled in the []().
 - To get started in developping a retraining pipeline on a separate Google Colab notebook, use []() as a starting point.<br />
There, you'll see how to connect the two. That way, you can run dev pipeline runs using the Google Colab free-tier GPUs !<br />
<em><small>(Note that this option is not advised for high CPU workloads, since clearly this is not the strong suit of Google Colab.)</small></em><br />
For details on the inner-workings of this <b>Statefull Metaflow Service+UI</b> installation, go read our dedicated article on HuggingFace @https://hf.co/.......... .

Note that, if unspecified while executing the `startup.sh` script, the default config for this setup is `/data` as MF_ROOT (to host <code>logs</code> and <code>persistent db</code> and <code>artifacts store</code>) and the exposed port number is `7860`. Just remember that this is a setup for which the host for <b>Metaflow</b> and the host for the runtime executing flow runs <u>shall share read/write access to the same MF_ROOT dir (same storage, same fullpath)</u>.

---

