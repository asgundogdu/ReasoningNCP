## Ray Cluster Quick Reference

This file contains the essential commands to **start**, **connect**, and **terminate** your one-node Ray cluster on your remote desktop (`shmooze`) and manage it from any Tailnet-joined device. It also shows how to use `uv` for Python package management and how to install necessary Ray libraries.

---

## Prerequisites: Python Package Management with `uv`

1. **Install `uv`** (an extremely fast Python package and project manager written in Rust):
	```bash
   	# Linux/macOS via install script
   	curl -LsSf https://astral.sh/uv/install.sh | sh
  	 ```

2. **Install Ray libraries using uv**:
   	```bash
  	# Install Ray core with dashboard and cluster launcher extras
	uv pip install "ray[default]"

	```

3. ** Start the Ray Head (shmooze)**:

	```
ray start --head \
  --node-ip-address=100.76.233.99 \
  --port=6379 \
  --include-dashboard=true \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --ray-client-server-port=10001 \
  --num-gpus=5	
```

Initializes the GCS server on port 6379.

Exposes the Dashboard on port 8265.

Opens Ray Client server on port 10001.


UI should be visable from http://shmooze.tail22f087.ts.net:8265/#/overview

4. **Check Cluster Status**:
```ray status
# For more detail:
ray status -v```

5. **Terminate the Cluster**:
```ray stop ```


