import requests
import os
import logging

log = logging.getLogger("uvicorn.error")

KNOX_URI = os.getenv("KNOX_URI", "http://localhost:8080")

class KnoxRequest:

    def update_job_status(self, job_id: str, status: str):
        url = f"{KNOX_URI}/job/status"
        payload = {
            "jobID": job_id,
            "status": status
        }
        log.info("Updating job status for job_id=%s to status=%s", job_id, status)
        try:
            response = requests.post(url, params=payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to update job status: {e}")
        
    