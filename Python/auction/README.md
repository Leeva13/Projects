# Auction Service

This is a FastAPI-based auction service with WebSocket support for real-time bid updates and time extensions. It uses PostgreSQL as the database and is containerized with Docker.

## Features
- Create auction lots via REST.
- Place bids on lots via REST.
- Get list of active lots.
- Subscribe to lot updates via WebSocket for real-time notifications on new bids and time extensions (if a bid is placed in the last minute, the auction extends by 1 minute).

## Project Structure
auction_service/
│
├── app/                  # The main folder with the code of our application
│   ├── __init__.py       # Makes 'app' a Python package
│   ├── main.py           # Main file with FastAPI and WebSocket logic
│   ├── models.py         # Database models (SQLAlchemy)
│   ├── schemas.py        # Schemes for data validation (Pydantic)
│   ├── database.py       # Setting up a connection to the database
│   └── crud.py           # Functions for working with the database (Create, Read, Update, Delete)
│
├── .gitignore            # File to ignore unnecessary Git files
├── Dockerfile            # Instructions for creating a Docker image
├── docker-compose.yml    # File to run the application and database together
├── requirements.txt      # List of project dependencies
└── README.md             # Project description and launch instructions

## How to Run
1. Clone the repository:
git clone <your-repo-url>
cd auction</your-repo-url>

2. Build and start the containers:
docker-compose up --build

3. The API will be available at `http://localhost:8000`. You can access the docs at `http://localhost:8000/docs`.

4. To test WebSocket, connect to `ws://localhost:8000/ws/lots/{lot_id}` (e.g., using a WebSocket client like wscat or Postman).

## Notes
- The database tables are automatically created on startup.
- Auctions end automatically based on `end_time`, with lazy status updates on access.
- Time extension: If a bid is placed within the last minute, the end time is extended by 1 minute, and a notification is sent via WebSocket.
- No authentication is implemented; bidders are identified by a string name.