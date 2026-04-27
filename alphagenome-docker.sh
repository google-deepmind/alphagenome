#!/bin/bash

# AlphaGenome Docker Compose Helper Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists and has API key
check_env() {
    if [ ! -f ".env" ]; then
        print_error ".env file not found!"
        echo "Please create a .env file with your ALPHAGENOME_API_KEY"
        exit 1
    fi
    
    if grep -q "{{YOUR_API_KEY}}" .env; then
        print_warning "Please update your ALPHAGENOME_API_KEY in the .env file"
    fi
}

# Main menu
show_menu() {
    echo "================================================"
    echo "          AlphaGenome Docker Manager"
    echo "================================================"
    echo "1. Start all services"
    echo "2. Stop all services"
    echo "3. View logs"
    echo "4. Open AlphaGenome shell"
    echo "5. Open Jupyter Lab"
    echo "6. Build/rebuild containers"
    echo "7. Clean up (remove containers and volumes)"
    echo "8. Show service status"
    echo "0. Exit"
    echo "================================================"
}

case "${1:-menu}" in
    "start"|"up")
        check_env
        print_status "Starting AlphaGenome services..."
        docker compose up -d
        print_status "Services started! Access points:"
        echo "  - Jupyter Lab: http://localhost:8889"
        echo "  - AlphaGenome container: docker compose exec alphagenome bash"
        echo "  - PostgreSQL: localhost:5432"
        echo "  - Redis: localhost:6379"
        ;;
    "stop"|"down")
        print_status "Stopping AlphaGenome services..."
        docker compose down
        ;;
    "logs")
        docker compose logs -f
        ;;
    "shell"|"bash")
        docker compose exec alphagenome bash
        ;;
    "jupyter")
        print_status "Opening Jupyter Lab..."
        open http://localhost:8889 2>/dev/null || echo "Visit http://localhost:8889"
        ;;
    "build")
        print_status "Building/rebuilding containers..."
        docker compose build --no-cache
        ;;
    "clean")
        print_warning "This will remove all containers and volumes!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker compose down -v --remove-orphans
            docker compose build --no-cache
            print_status "Cleanup complete!"
        fi
        ;;
    "status")
        docker compose ps
        ;;
    "menu"|*)
        while true; do
            show_menu
            read -p "Select an option [0-8]: " choice
            case $choice in
                1) $0 start ;;
                2) $0 stop ;;
                3) $0 logs ;;
                4) $0 shell ;;
                5) $0 jupyter ;;
                6) $0 build ;;
                7) $0 clean ;;
                8) $0 status ;;
                0) exit 0 ;;
                *) print_error "Invalid option. Please try again." ;;
            esac
            echo
            read -p "Press Enter to continue..."
        done
        ;;
esac