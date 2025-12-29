
echo "Starting Movie Recommender Setup..."

if ! command -v python3 &> /dev/null
then
    echo "Python3 not found. Please install Python3."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo " Upgrading pip..."
pip install --upgrade pip

echo " Installing dependencies..."
pip install -r requirements.txt

if ! command -v redis-server &> /dev/null
then
    echo " Redis not found. Please install Redis."
    exit 1
fi

echo " Starting Redis..."
redis-server --daemonize yes

echo "Running recommender..."
python recommender.py
