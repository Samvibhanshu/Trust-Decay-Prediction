"""python run.py  starts the dashboard on http://127.0.0.1:5000  (set TRUSTDECAY_API_KEY to protect POST endpoints)."""
import os

from trustdecay.webapp import create_app

if __name__ == "__main__":
    create_app(db_path=os.environ.get("TRUSTDECAY_DB", "trustdecay.db")).run(port=int(os.environ.get("PORT", 5000)))
