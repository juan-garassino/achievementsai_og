#!/bin/bash
# Add NebulaGraph Studio to your setup

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}Adding NebulaGraph Studio to your setup...${NC}"

# Check if in the nebula-docker directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found.${NC}"
    echo -e "${YELLOW}Please run this script from your nebula-docker directory.${NC}"
    exit 1
fi

# Create updated docker-compose with Studio
echo -e "${CYAN}Updating docker-compose.yml to include Studio...${NC}"

# Backup original docker-compose.yml
cp docker-compose.yml docker-compose.yml.bak

# Create updated docker-compose.yml with Studio added
cat > docker-compose.yml << 'EOF'
services:
  metad:
    image: vesoft/nebula-metad:v3.5.0
    environment:
      - TZ=UTC
      - USER=root
    command:
      - --meta_server_addrs=172.28.1.1:9559
      - --local_ip=172.28.1.1
      - --ws_ip=172.28.1.1
      - --port=9559
      - --ws_http_port=19559
      - --data_path=/data/meta
      - --log_dir=/logs
      - --v=0
      - --minloglevel=0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://172.28.1.1:19559/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    ports:
      - "9559:9559"
      - "19559:19559"
    volumes:
      - ./data/meta:/data/meta
      - ./logs:/logs
    networks:
      nebula-net:
        ipv4_address: 172.28.1.1
    restart: on-failure
    cap_add:
      - SYS_PTRACE

  storaged:
    image: vesoft/nebula-storaged:v3.5.0
    environment:
      - TZ=UTC
      - USER=root
    command:
      - --meta_server_addrs=172.28.1.1:9559
      - --local_ip=172.28.1.2
      - --ws_ip=172.28.1.2
      - --port=9779
      - --ws_http_port=19779
      - --data_path=/data/storage
      - --log_dir=/logs
      - --v=0
      - --minloglevel=0
    depends_on:
      - metad
    healthcheck:
      test: ["CMD", "curl", "-f", "http://172.28.1.2:19779/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    ports:
      - "9779:9779"
      - "19779:19779"
    volumes:
      - ./data/storage0:/data/storage
      - ./logs:/logs
    networks:
      nebula-net:
        ipv4_address: 172.28.1.2
    restart: on-failure
    cap_add:
      - SYS_PTRACE

  graphd:
    image: vesoft/nebula-graphd:v3.5.0
    environment:
      - TZ=UTC
      - USER=root
    command:
      - --meta_server_addrs=172.28.1.1:9559
      - --local_ip=172.28.1.3
      - --ws_ip=172.28.1.3
      - --port=9669
      - --ws_http_port=19669
      - --log_dir=/logs
      - --v=0
      - --minloglevel=0
    depends_on:
      - metad
    healthcheck:
      test: ["CMD", "curl", "-f", "http://172.28.1.3:19669/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
    ports:
      - "9669:9669"
      - "19669:19669"
    volumes:
      - ./logs:/logs
    networks:
      nebula-net:
        ipv4_address: 172.28.1.3
    restart: on-failure
    cap_add:
      - SYS_PTRACE

  console:
    image: vesoft/nebula-console:v3.5.0
    entrypoint: ""
    command: tail -f /dev/null
    depends_on:
      - graphd
    networks:
      - nebula-net

  studio:
    image: vesoft/nebula-graph-studio:v3.5.0
    environment:
      - TZ=UTC
    ports:
      - "7001:7001"
    networks:
      - nebula-net
    depends_on:
      - graphd
    restart: on-failure

networks:
  nebula-net:
    ipam:
      driver: default
      config:
        - subnet: 172.28.0.0/16
EOF

# Restart the containers
echo -e "${YELLOW}Restarting containers to apply changes...${NC}"
docker-compose down
docker-compose up -d

# Wait for services to restart
echo -e "${YELLOW}Waiting for services to restart (30s)...${NC}"
sleep 30

# Check if Studio is running
STUDIO_RUNNING=$(docker-compose ps studio | grep "Up")
if [[ ! -z "$STUDIO_RUNNING" ]]; then
    echo -e "${GREEN}NebulaGraph Studio is now running!${NC}"
    echo -e "${CYAN}Access the Studio interface at:${NC}"
    echo -e "${YELLOW}http://localhost:7001${NC}"
    echo -e ""
    echo -e "${CYAN}Connection settings to use in Studio:${NC}"
    echo -e "${YELLOW}Host: graphd${NC}"
    echo -e "${YELLOW}Port: 9669${NC}"
    echo -e "${YELLOW}Username: root${NC}"
    echo -e "${YELLOW}Password: nebula${NC}"
else
    echo -e "${RED}NebulaGraph Studio did not start properly.${NC}"
    echo -e "${YELLOW}Please check the logs with: docker-compose logs studio${NC}"
fi

echo -e "${CYAN}Don't forget to run these commands to prepare your database:${NC}"
echo -e "${YELLOW}docker-compose exec console nebula-console -addr graphd -port 9669 -u root -p nebula -e \"ADD HOSTS '172.28.1.2':9779;\"${NC}"
echo -e "${YELLOW}docker-compose exec console nebula-console -addr graphd -port 9669 -u root -p nebula -e \"CREATE SPACE IF NOT EXISTS llamaindex_property_graph(vid_type=FIXED_STRING(256), replica_factor=1, partition_num=1);\"${NC}"