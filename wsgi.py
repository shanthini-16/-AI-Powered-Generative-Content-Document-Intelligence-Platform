"""
WSGI Production Entry Point for Render & Gunicorn Deployment
"""

import os
from web.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
