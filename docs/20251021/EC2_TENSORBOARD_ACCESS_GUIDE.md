# EC2 上執行 FreqAI 訓練時存取 TensorBoard 指南

**日期**: 2025年10月21日  
**目的**: 在 AWS EC2 執行 FreqAI PPO 訓練時，從本地機器存取 TensorBoard 監控介面

---

## 概述

當您在 AWS EC2 實例上訓練 FreqAI RL 模型時，TensorBoard 提供即時的訓練監控。本指南涵蓋所有存取 TensorBoard 的方法，從最安全到最簡單。

### TensorBoard 在 FreqAI 中的位置

根據您的配置 (`config_v1.json`)：
- **Identifier**: `PPO_15m_v1`
- **TensorBoard 日誌路徑**: `user_data/models/PPO_15m_v1/sub-train-{strategy_name}/tensorboard/`
- **預設端口**: 6006

---

## 方法比較

| 方法 | 安全性 | 難度 | 速度 | 推薦度 |
|------|--------|------|------|--------|
| SSH 端口轉發 | ⭐⭐⭐⭐⭐ | 簡單 | 快 | ⭐⭐⭐⭐⭐ 最推薦 |
| AWS Session Manager | ⭐⭐⭐⭐⭐ | 中等 | 快 | ⭐⭐⭐⭐☆ |
| 安全組 + 公開存取 | ⭐⭐☆☆☆ | 最簡單 | 快 | ⭐⭐☆☆☆ 不推薦 |
| Ngrok 隧道 | ⭐⭐⭐☆☆ | 簡單 | 中等 | ⭐⭐⭐☆☆ |

---

## 方法 1：SSH 端口轉發（最推薦）⭐

### 優點
- ✅ 最安全（加密連線）
- ✅ 不需要修改安全組
- ✅ 不暴露服務到公網
- ✅ 設置簡單
- ✅ 免費

### 步驟 1：在 EC2 上啟動 TensorBoard

```bash
# SSH 連線到 EC2
ssh -i your-key.pem ubuntu@ec2-instance-ip

# 進入專案目錄
cd ~/freqai_training

# 啟動虛擬環境
source .venv/bin/activate

# 啟動 TensorBoard（背景執行）
nohup tensorboard --logdir user_data/models/PPO_15m_v1 \
  --host 127.0.0.1 \
  --port 6006 \
  > tensorboard.log 2>&1 &

# 檢查是否運行
ps aux | grep tensorboard

# 查看日誌
tail -f tensorboard.log
```

### 步驟 2：設置 SSH 端口轉發（本地機器）

#### Windows PowerShell：
```powershell
# 基本端口轉發
ssh -i "path\to\your-key.pem" -L 6006:localhost:6006 ubuntu@ec2-instance-ip

# 或者在背景執行
Start-Process ssh -ArgumentList "-i", "path\to\your-key.pem", "-L", "6006:localhost:6006", "-N", "-f", "ubuntu@ec2-instance-ip"
```

#### Linux/Mac：
```bash
# 基本端口轉發
ssh -i your-key.pem -L 6006:localhost:6006 ubuntu@ec2-instance-ip

# 背景執行
ssh -i your-key.pem -L 6006:localhost:6006 -N -f ubuntu@ec2-instance-ip
```

**參數說明**：
- `-L 6006:localhost:6006`：將本地 6006 端口轉發到遠端的 6006 端口
- `-N`：不執行遠端命令（僅端口轉發）
- `-f`：背景執行

### 步驟 3：存取 TensorBoard

在本地瀏覽器開啟：
```
http://localhost:6006
```

### 步驟 4：關閉連線（完成後）

```bash
# 找到 SSH 進程
# Windows PowerShell:
Get-Process ssh

# Linux/Mac:
ps aux | grep ssh

# 終止進程
# Windows:
Stop-Process -Id <PID>

# Linux/Mac:
kill <PID>
```

---

## 方法 2：AWS Session Manager 端口轉發

### 優點
- ✅ 非常安全
- ✅ 不需要 SSH 密鑰
- ✅ 不需要公開 IP
- ✅ 通過 AWS IAM 控制存取

### 先決條件

1. EC2 實例需要 SSM Agent（Amazon Linux 2/Ubuntu 18.04+ 預裝）
2. EC2 實例角色需要 `AmazonSSMManagedInstanceCore` 政策
3. 本地安裝 AWS CLI 和 Session Manager 插件

### 安裝 Session Manager 插件

#### Windows：
```powershell
# 下載並安裝
# https://s3.amazonaws.com/session-manager-downloads/plugin/latest/windows/SessionManagerPluginSetup.exe
```

#### Linux：
```bash
curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o "session-manager-plugin.deb"
sudo dpkg -i session-manager-plugin.deb
```

#### Mac：
```bash
brew install --cask session-manager-plugin
```

### 步驟 1：為 EC2 實例添加 IAM 角色

```bash
# 建立信任政策
cat > ec2-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# 建立角色
aws iam create-role \
  --role-name EC2-SSM-Role \
  --assume-role-policy-document file://ec2-trust-policy.json

# 附加 SSM 政策
aws iam attach-role-policy \
  --role-name EC2-SSM-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# 建立實例配置檔
aws iam create-instance-profile --instance-profile-name EC2-SSM-Profile

# 將角色添加到配置檔
aws iam add-role-to-instance-profile \
  --instance-profile-name EC2-SSM-Profile \
  --role-name EC2-SSM-Role

# 將配置檔附加到 EC2 實例
aws ec2 associate-iam-instance-profile \
  --instance-id i-1234567890abcdef0 \
  --iam-instance-profile Name=EC2-SSM-Profile
```

### 步驟 2：啟動 TensorBoard（EC2 上）

```bash
# 連線到 EC2（使用 Session Manager）
aws ssm start-session --target i-1234567890abcdef0

# 在 EC2 上
cd ~/freqai_training
source .venv/bin/activate
nohup tensorboard --logdir user_data/models/PPO_15m_v1 --host 127.0.0.1 --port 6006 > tensorboard.log 2>&1 &
exit
```

### 步驟 3：設置端口轉發（本地機器）

```bash
# 轉發 TensorBoard 端口
aws ssm start-session \
  --target i-1234567890abcdef0 \
  --document-name AWS-StartPortForwardingSession \
  --parameters "portNumber=6006,localPortNumber=6006"
```

### 步驟 4：存取 TensorBoard

在瀏覽器開啟：
```
http://localhost:6006
```

---

## 方法 3：安全組 + 公開存取（不推薦）

### ⚠️ 警告
此方法將 TensorBoard 暴露到公網，**強烈不推薦**用於生產環境。僅適用於：
- 短期測試
- 臨時演示
- 您了解安全風險

### 步驟 1：修改安全組

```bash
# 獲取您的公網 IP
MY_IP=$(curl -s https://checkip.amazonaws.com)

# 添加入站規則（僅允許您的 IP）
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp \
  --port 6006 \
  --cidr ${MY_IP}/32 \
  --description "TensorBoard access from my IP"
```

或在 AWS Console：
1. EC2 → Security Groups
2. 選擇您的安全組
3. Inbound rules → Edit inbound rules
4. Add rule:
   - Type: Custom TCP
   - Port: 6006
   - Source: My IP（或特定 IP/32）

### 步驟 2：啟動 TensorBoard（綁定到所有介面）

```bash
# SSH 到 EC2
ssh -i your-key.pem ubuntu@ec2-instance-ip

# 啟動 TensorBoard（綁定到 0.0.0.0）
cd ~/freqai_training
source .venv/bin/activate
nohup tensorboard --logdir user_data/models/PPO_15m_v1 \
  --host 0.0.0.0 \
  --port 6006 \
  > tensorboard.log 2>&1 &
```

### 步驟 3：存取 TensorBoard

在瀏覽器開啟：
```
http://ec2-public-ip:6006
```

### 步驟 4：完成後刪除規則（重要！）

```bash
# 刪除入站規則
aws ec2 revoke-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp \
  --port 6006 \
  --cidr ${MY_IP}/32
```

---

## 方法 4：Ngrok 隧道

### 優點
- ✅ 設置簡單
- ✅ 提供 HTTPS
- ✅ 無需修改安全組
- ✅ 可分享連結

### 缺點
- ❌ 免費版有限制
- ❌ 依賴第三方服務
- ❌ 可能有延遲

### 步驟 1：安裝 Ngrok（EC2 上）

```bash
# SSH 到 EC2
ssh -i your-key.pem ubuntu@ec2-instance-ip

# 下載 Ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# 或使用 snap
sudo snap install ngrok
```

### 步驟 2：配置 Ngrok

```bash
# 註冊 Ngrok 帳戶：https://dashboard.ngrok.com/signup
# 獲取 authtoken

# 設置 authtoken
ngrok config add-authtoken YOUR_AUTHTOKEN
```

### 步驟 3：啟動 TensorBoard 和 Ngrok

```bash
# 啟動 TensorBoard
cd ~/freqai_training
source .venv/bin/activate
nohup tensorboard --logdir user_data/models/PPO_15m_v1 --host 127.0.0.1 --port 6006 > tensorboard.log 2>&1 &

# 啟動 Ngrok
ngrok http 6006
```

### 步驟 4：存取 TensorBoard

Ngrok 會顯示公開 URL：
```
Forwarding    https://xxxx-xx-xx-xxx-xxx.ngrok-free.app -> http://localhost:6006
```

在瀏覽器開啟該 URL。

---

## 自動化腳本

### 完整啟動腳本（EC2 上）

建立 `start_tensorboard.sh`：

```bash
#!/bin/bash

# FreqAI TensorBoard 啟動腳本
echo "=== Starting TensorBoard for FreqAI Training ==="

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
PROJECT_DIR="$HOME/freqai_training"
VENV_PATH="$PROJECT_DIR/.venv"
TENSORBOARD_PORT=6006
LOG_FILE="$PROJECT_DIR/tensorboard.log"

# FreqAI 配置
FREQAI_IDENTIFIER="PPO_15m_v1"
TENSORBOARD_LOGDIR="$PROJECT_DIR/user_data/models/$FREQAI_IDENTIFIER"

# 檢查虛擬環境
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3.11 -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    pip install tensorboard
else
    source $VENV_PATH/bin/activate
fi

# 檢查 TensorBoard 是否已經運行
if pgrep -f "tensorboard.*$TENSORBOARD_PORT" > /dev/null; then
    echo -e "${YELLOW}TensorBoard is already running on port $TENSORBOARD_PORT${NC}"
    echo "PID: $(pgrep -f "tensorboard.*$TENSORBOARD_PORT")"
    read -p "Kill and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "tensorboard.*$TENSORBOARD_PORT"
        echo "Killed existing TensorBoard process"
        sleep 2
    else
        echo "Exiting..."
        exit 0
    fi
fi

# 檢查日誌目錄是否存在
if [ ! -d "$TENSORBOARD_LOGDIR" ]; then
    echo -e "${YELLOW}Warning: TensorBoard log directory does not exist yet:${NC}"
    echo "$TENSORBOARD_LOGDIR"
    echo "TensorBoard will wait for logs to be created during training."
    echo ""
fi

# 啟動 TensorBoard
echo -e "${GREEN}Starting TensorBoard...${NC}"
echo "Log directory: $TENSORBOARD_LOGDIR"
echo "Port: $TENSORBOARD_PORT"
echo "Log file: $LOG_FILE"
echo ""

nohup tensorboard \
    --logdir "$TENSORBOARD_LOGDIR" \
    --host 127.0.0.1 \
    --port $TENSORBOARD_PORT \
    --reload_interval 30 \
    > "$LOG_FILE" 2>&1 &

TENSORBOARD_PID=$!

# 等待啟動
sleep 3

# 驗證
if ps -p $TENSORBOARD_PID > /dev/null; then
    echo -e "${GREEN}✓ TensorBoard started successfully!${NC}"
    echo "PID: $TENSORBOARD_PID"
    echo ""
    echo "Access TensorBoard:"
    echo "  - Via SSH tunnel: http://localhost:$TENSORBOARD_PORT"
    echo "  - View logs: tail -f $LOG_FILE"
    echo ""
    echo "To stop TensorBoard:"
    echo "  kill $TENSORBOARD_PID"
    echo "  or"
    echo "  pkill -f 'tensorboard.*$TENSORBOARD_PORT'"
else
    echo -e "${YELLOW}✗ Failed to start TensorBoard${NC}"
    echo "Check logs: cat $LOG_FILE"
    exit 1
fi
```

使用方式：

```bash
# 賦予執行權限
chmod +x start_tensorboard.sh

# 執行
./start_tensorboard.sh
```

### 停止 TensorBoard 腳本

建立 `stop_tensorboard.sh`：

```bash
#!/bin/bash

echo "=== Stopping TensorBoard ==="

# 找到並終止 TensorBoard 進程
if pgrep -f "tensorboard" > /dev/null; then
    pkill -f "tensorboard"
    echo "✓ TensorBoard stopped"
else
    echo "✗ TensorBoard is not running"
fi
```

### SSH 端口轉發腳本（本地 Windows PowerShell）

建立 `Connect-TensorBoard.ps1`：

```powershell
# FreqAI TensorBoard SSH Tunnel Script

param(
    [Parameter(Mandatory=$true)]
    [string]$KeyPath,
    
    [Parameter(Mandatory=$true)]
    [string]$EC2Host,
    
    [int]$LocalPort = 6006,
    [int]$RemotePort = 6006,
    [string]$Username = "ubuntu"
)

Write-Host "=== FreqAI TensorBoard SSH Tunnel ===" -ForegroundColor Cyan

# 檢查密鑰檔案
if (-not (Test-Path $KeyPath)) {
    Write-Host "Error: SSH key not found at $KeyPath" -ForegroundColor Red
    exit 1
}

# 檢查端口是否已被佔用
$portInUse = Get-NetTCPConnection -LocalPort $LocalPort -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "Warning: Port $LocalPort is already in use" -ForegroundColor Yellow
    Write-Host "Current connections on port ${LocalPort}:" -ForegroundColor Yellow
    $portInUse | Format-Table -AutoSize
    
    $response = Read-Host "Kill existing SSH tunnels and continue? (y/n)"
    if ($response -eq 'y') {
        Get-Process ssh -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep -Seconds 2
    } else {
        exit 0
    }
}

# 建立 SSH 隧道
Write-Host "Creating SSH tunnel..." -ForegroundColor Green
Write-Host "Local: localhost:$LocalPort -> Remote: $EC2Host:$RemotePort" -ForegroundColor Green

$sshArgs = @(
    "-i", $KeyPath,
    "-L", "${LocalPort}:localhost:${RemotePort}",
    "-N",
    "${Username}@${EC2Host}"
)

Write-Host ""
Write-Host "Starting SSH tunnel (press Ctrl+C to stop)..." -ForegroundColor Yellow
Write-Host "Once connected, open browser: http://localhost:$LocalPort" -ForegroundColor Cyan
Write-Host ""

# 啟動 SSH（前景執行，可以用 Ctrl+C 終止）
& ssh $sshArgs

Write-Host ""
Write-Host "SSH tunnel closed" -ForegroundColor Yellow
```

使用方式：

```powershell
# 執行腳本
.\Connect-TensorBoard.ps1 -KeyPath "C:\path\to\key.pem" -EC2Host "ec2-xx-xx-xx-xx.compute.amazonaws.com"

# 或指定用戶名
.\Connect-TensorBoard.ps1 -KeyPath "C:\path\to\key.pem" -EC2Host "1.2.3.4" -Username "ubuntu"
```

### SSH 端口轉發腳本（本地 Linux/Mac）

建立 `connect_tensorboard.sh`：

```bash
#!/bin/bash

# FreqAI TensorBoard SSH Tunnel Script

usage() {
    echo "Usage: $0 -k <key-path> -h <ec2-host> [-l <local-port>] [-r <remote-port>] [-u <username>]"
    echo "  -k: Path to SSH private key (required)"
    echo "  -h: EC2 host/IP (required)"
    echo "  -l: Local port (default: 6006)"
    echo "  -r: Remote port (default: 6006)"
    echo "  -u: SSH username (default: ubuntu)"
    exit 1
}

# 預設值
LOCAL_PORT=6006
REMOTE_PORT=6006
USERNAME="ubuntu"

# 解析參數
while getopts "k:h:l:r:u:" opt; do
    case $opt in
        k) KEY_PATH="$OPTARG" ;;
        h) EC2_HOST="$OPTARG" ;;
        l) LOCAL_PORT="$OPTARG" ;;
        r) REMOTE_PORT="$OPTARG" ;;
        u) USERNAME="$OPTARG" ;;
        *) usage ;;
    esac
done

# 檢查必要參數
if [ -z "$KEY_PATH" ] || [ -z "$EC2_HOST" ]; then
    usage
fi

# 檢查密鑰檔案
if [ ! -f "$KEY_PATH" ]; then
    echo "Error: SSH key not found at $KEY_PATH"
    exit 1
fi

echo "=== FreqAI TensorBoard SSH Tunnel ==="
echo "Key: $KEY_PATH"
echo "Host: $EC2_HOST"
echo "Tunnel: localhost:$LOCAL_PORT -> $EC2_HOST:$REMOTE_PORT"
echo ""

# 檢查端口是否已被佔用
if lsof -Pi :$LOCAL_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Warning: Port $LOCAL_PORT is already in use"
    echo "Current process:"
    lsof -Pi :$LOCAL_PORT -sTCP:LISTEN
    echo ""
    read -p "Kill existing SSH tunnels and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "ssh.*-L.*$LOCAL_PORT"
        sleep 2
    else
        exit 0
    fi
fi

echo "Starting SSH tunnel (press Ctrl+C to stop)..."
echo "Once connected, open browser: http://localhost:$LOCAL_PORT"
echo ""

# 建立 SSH 隧道
ssh -i "$KEY_PATH" \
    -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} \
    -N \
    ${USERNAME}@${EC2_HOST}

echo ""
echo "SSH tunnel closed"
```

使用方式：

```bash
# 賦予執行權限
chmod +x connect_tensorboard.sh

# 執行
./connect_tensorboard.sh -k ~/.ssh/your-key.pem -h ec2-xx-xx-xx-xx.compute.amazonaws.com

# 或自訂端口
./connect_tensorboard.sh -k ~/.ssh/your-key.pem -h 1.2.3.4 -l 6006 -r 6006
```

---

## FreqAI 特定配置

### 更新配置以啟用 GPU

在部署到 GPU EC2 實例之前，更新 `config_v1.json`：

```json
{
  "freqai": {
    "model_training_parameters": {
      "device": "cuda",  // 從 "cpu" 改為 "cuda"
      ...
    }
  }
}
```

### TensorBoard 監控指標

FreqAI PPO 訓練會記錄以下指標到 TensorBoard：

#### 訓練指標
- **train/loss**: 總損失
- **train/policy_loss**: 策略損失
- **train/value_loss**: 價值函數損失
- **train/entropy_loss**: 熵損失
- **train/learning_rate**: 學習率
- **train/clip_fraction**: 裁剪分數

#### 回報指標
- **rollout/ep_rew_mean**: 平均回報
- **rollout/ep_len_mean**: 平均回合長度

#### 時間指標
- **time/fps**: 每秒幀數
- **time/iterations**: 迭代次數

### 訓練期間啟動 TensorBoard

```bash
# 在一個終端啟動訓練
cd ~/freqai_training
source .venv/bin/activate

freqtrade backtesting \
  --strategy RLStrategy4Action \
  --config user_data/config_v1.json \
  --freqaimodel ReinforcementLearner4Action \
  --timerange 20230101-20251019

# 在另一個終端（或使用 tmux/screen）啟動 TensorBoard
./start_tensorboard.sh
```

### 使用 tmux 同時運行訓練和 TensorBoard

```bash
# 安裝 tmux
sudo apt install tmux -y

# 建立 tmux 會話
tmux new-session -d -s freqai

# 窗口 0：訓練
tmux send-keys -t freqai:0 "cd ~/freqai_training && source .venv/bin/activate" C-m
tmux send-keys -t freqai:0 "freqtrade backtesting --strategy RLStrategy4Action --config user_data/config_v1.json --freqaimodel ReinforcementLearner4Action --timerange 20230101-20251019" C-m

# 建立窗口 1：TensorBoard
tmux new-window -t freqai:1 -n tensorboard
tmux send-keys -t freqai:1 "cd ~/freqai_training && ./start_tensorboard.sh" C-m

# 附加到會話
tmux attach -t freqai

# 快捷鍵：
# Ctrl+B, 0 - 切換到訓練窗口
# Ctrl+B, 1 - 切換到 TensorBoard 窗口
# Ctrl+B, D - 分離會話（訓練繼續在背景執行）

# 重新附加到會話
tmux attach -t freqai
```

---

## 完整部署工作流程

### 步驟 1：準備 EC2 實例

```bash
# 1. 啟動 EC2 實例（使用 Deep Learning AMI）
# 參考：AWS_SPOT_GPU_ANALYSIS.md

# 2. SSH 連線
ssh -i your-key.pem ubuntu@ec2-ip

# 3. 設置 FreqAI 環境
git clone https://github.com/your-repo/freqtrade.git
cd freqtrade
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install freqtrade[freqai-rl]

# 4. 複製配置檔案
# 上傳 config_v1.json, RLStrategy4Action.py 等
```

### 步驟 2：設置 TensorBoard

```bash
# 建立啟動腳本
cat > start_tensorboard.sh << 'EOF'
[上面的完整腳本內容]
EOF

chmod +x start_tensorboard.sh
```

### 步驟 3：啟動訓練和監控

```bash
# 使用 tmux
tmux new -s freqai

# 在 tmux 中啟動訓練
source .venv/bin/activate
freqtrade backtesting \
  --strategy RLStrategy4Action \
  --config user_data/config_v1.json \
  --freqaimodel ReinforcementLearner4Action \
  --timerange 20230101-20251019

# 建立新窗口（Ctrl+B, C）
./start_tensorboard.sh

# 分離 tmux（Ctrl+B, D）
```

### 步驟 4：本地存取 TensorBoard

```powershell
# Windows PowerShell（本地機器）
.\Connect-TensorBoard.ps1 -KeyPath "C:\path\to\key.pem" -EC2Host "ec2-ip"
```

### 步驟 5：開啟瀏覽器

```
http://localhost:6006
```

---

## 疑難排解

### 問題 1：TensorBoard 顯示 "No dashboards are active"

**原因**：訓練尚未開始產生日誌，或日誌目錄錯誤

**解決方案**：
```bash
# 檢查日誌目錄
ls -la user_data/models/PPO_15m_v1/

# 確認 identifier 是否正確
grep identifier user_data/config_v1.json

# 等待訓練開始（通常需要幾分鐘）
```

---

### 問題 2：SSH 端口轉發連線被拒絕

**原因**：TensorBoard 未運行，或綁定到錯誤的介面

**解決方案**：
```bash
# 在 EC2 上檢查 TensorBoard
ps aux | grep tensorboard

# 檢查端口
netstat -tlnp | grep 6006

# 查看 TensorBoard 日誌
tail -f tensorboard.log

# 確保綁定到 127.0.0.1
# tensorboard --host 127.0.0.1 --port 6006
```

---

### 問題 3：本地端口已被佔用

**Windows PowerShell**：
```powershell
# 找到佔用端口的進程
Get-NetTCPConnection -LocalPort 6006 | Format-Table -AutoSize

# 終止進程
Stop-Process -Id <PID>
```

**Linux/Mac**：
```bash
# 找到進程
lsof -i :6006

# 終止進程
kill <PID>
```

---

### 問題 4：TensorBoard 很慢或無回應

**原因**：日誌檔案太大，或網路延遲

**解決方案**：
```bash
# 1. 減少重新載入頻率
tensorboard --logdir user_data/models/PPO_15m_v1 --reload_interval 60

# 2. 只載入最近的資料
tensorboard --logdir user_data/models/PPO_15m_v1 --reload_multifile=true --reload_multifile_inactive_secs=300

# 3. 壓縮舊日誌
cd user_data/models/PPO_15m_v1
find . -name "*.tfevents.*" -mtime +7 -exec gzip {} \;
```

---

### 問題 5：無法連線到 EC2

**檢查清單**：
```bash
# 1. EC2 實例是否運行
aws ec2 describe-instances --instance-ids i-xxxxx

# 2. 安全組是否允許 SSH（端口 22）
aws ec2 describe-security-groups --group-ids sg-xxxxx

# 3. 密鑰權限
# Windows: 檔案屬性 > 安全性 > 只給自己完全控制
# Linux/Mac:
chmod 400 your-key.pem

# 4. 測試連線
ssh -v -i your-key.pem ubuntu@ec2-ip
```

---

### 問題 6：TensorBoard 顯示舊資料

**原因**：快取問題

**解決方案**：
```bash
# 1. 強制重新整理瀏覽器（Ctrl+Shift+R）

# 2. 重啟 TensorBoard
pkill -f tensorboard
./start_tensorboard.sh

# 3. 清除 TensorBoard 快取
rm -rf /tmp/.tensorboard-info/
```

---

## 安全最佳實踐

### 1. 使用 SSH 端口轉發（不要公開暴露）

✅ **推薦**：
```bash
ssh -L 6006:localhost:6006 ubuntu@ec2-ip
tensorboard --host 127.0.0.1
```

❌ **避免**：
```bash
tensorboard --host 0.0.0.0  # 不要這樣做！
```

### 2. 限制安全組規則

如果必須公開存取：
```bash
# 只允許您的 IP
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 6006 \
  --cidr $(curl -s https://checkip.amazonaws.com)/32
```

### 3. 使用 IAM 角色（不要硬編碼憑證）

```bash
# 不要在程式碼中硬編碼 AWS 憑證
# 使用 IAM 角色和實例配置檔
```

### 4. 啟用 CloudWatch 日誌

```bash
# 安裝 CloudWatch Agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# 配置日誌收集
# 將 TensorBoard 和訓練日誌發送到 CloudWatch
```

### 5. 定期輪換 SSH 密鑰

```bash
# 定期更換 EC2 密鑰對
aws ec2 create-key-pair --key-name new-key --query 'KeyMaterial' --output text > new-key.pem
```

---

## 高級技巧

### 1. 比較多個訓練執行

```bash
# 啟動 TensorBoard 比較多個模型
tensorboard --logdir_spec=\
ppo_v1:user_data/models/PPO_15m_v1,\
ppo_v2:user_data/models/PPO_15m_v2,\
ppo_v3:user_data/models/PPO_15m_v3
```

### 2. 自動保存 TensorBoard 截圖

建立 `capture_tensorboard.py`：

```python
#!/usr/bin/env python3
"""
自動捕獲 TensorBoard 截圖
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from datetime import datetime

def capture_tensorboard(url="http://localhost:6006", output_dir="screenshots"):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    
    time.sleep(5)  # 等待頁面載入
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/tensorboard_{timestamp}.png"
    driver.save_screenshot(filename)
    
    driver.quit()
    print(f"Screenshot saved: {filename}")

if __name__ == "__main__":
    capture_tensorboard()
```

### 3. 自動化報告生成

```bash
# 使用 tensorboard 匯出資料
tensorboard --logdir user_data/models/PPO_15m_v1 --export_to_json report.json

# 或使用 Python API
python << EOF
from tensorboard.backend.event_processing import event_accumulator
import json

ea = event_accumulator.EventAccumulator('user_data/models/PPO_15m_v1')
ea.Reload()

# 提取訓練損失
train_loss = ea.Scalars('train/loss')
with open('training_metrics.json', 'w') as f:
    json.dump(train_loss, f, indent=2)

print("Metrics exported to training_metrics.json")
EOF
```

### 4. 設置警報

建立 `monitor_training.py`：

```python
#!/usr/bin/env python3
"""
監控訓練並發送警報
"""
import time
import requests
from tensorboard.backend.event_processing import event_accumulator

def check_training_health(logdir, threshold=10.0):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    
    if 'train/loss' in ea.Tags()['scalars']:
        loss = ea.Scalars('train/loss')
        if loss and loss[-1].value > threshold:
            send_alert(f"Training loss too high: {loss[-1].value}")
    
def send_alert(message):
    # 發送到 Slack、Email 等
    print(f"ALERT: {message}")
    # webhook_url = "YOUR_SLACK_WEBHOOK"
    # requests.post(webhook_url, json={"text": message})

if __name__ == "__main__":
    while True:
        check_training_health('user_data/models/PPO_15m_v1')
        time.sleep(300)  # 每 5 分鐘檢查一次
```

---

## 成本優化

### 1. 僅在需要時連線

```bash
# 不要持續保持 SSH 隧道
# 只在需要檢查訓練時連線

# 使用腳本快速連線和斷開
./connect_tensorboard.sh -k key.pem -h ec2-ip
# 檢查完畢後 Ctrl+C 斷開
```

### 2. 使用 Spot 實例

```bash
# Spot 實例更便宜（參考 AWS_SPOT_GPU_ANALYSIS.md）
# 確保實現檢查點以應對中斷
```

### 3. 定期清理舊日誌

```bash
# 建立清理腳本
cat > cleanup_logs.sh << 'EOF'
#!/bin/bash
# 刪除 30 天前的 TensorBoard 日誌
find user_data/models/*/tensorboard -name "*.tfevents.*" -mtime +30 -delete
echo "Old logs cleaned"
EOF

chmod +x cleanup_logs.sh

# 設置 cron job
crontab -e
# 添加：0 2 * * 0 /path/to/cleanup_logs.sh
```

---

## 快速參考

### 常用命令

```bash
# 啟動 TensorBoard
tensorboard --logdir user_data/models/PPO_15m_v1 --host 127.0.0.1 --port 6006

# 查看運行的 TensorBoard
ps aux | grep tensorboard

# 停止 TensorBoard
pkill -f tensorboard

# SSH 端口轉發
ssh -i key.pem -L 6006:localhost:6006 -N ubuntu@ec2-ip

# 檢查端口
netstat -tlnp | grep 6006  # Linux
netstat -an | findstr 6006  # Windows
```

### 故障排除快速檢查

```bash
# 1. TensorBoard 是否運行？
ps aux | grep tensorboard

# 2. 端口是否開放？
netstat -tlnp | grep 6006

# 3. 日誌目錄是否存在？
ls -la user_data/models/PPO_15m_v1/

# 4. SSH 隧道是否建立？
ps aux | grep "ssh.*6006"  # Linux
Get-Process ssh  # Windows

# 5. 查看 TensorBoard 日誌
tail -f tensorboard.log
```

---

## 總結

### 推薦方法

對於 FreqAI 訓練監控，推薦使用：

1. **SSH 端口轉發**（最安全、最簡單）⭐⭐⭐⭐⭐
   - 使用提供的 `Connect-TensorBoard.ps1` 或 `connect_tensorboard.sh`
   - 安全、快速、免費

2. **AWS Session Manager**（如果不想管理 SSH 密鑰）⭐⭐⭐⭐☆
   - 更安全
   - 需要額外設置

3. **避免公開暴露**（除非絕對必要）⭐⭐☆☆☆

### 關鍵要點

- ✅ 始終使用 SSH 端口轉發存取 TensorBoard
- ✅ 在 EC2 上使用 `--host 127.0.0.1` 啟動 TensorBoard
- ✅ 使用 tmux/screen 在背景運行訓練和 TensorBoard
- ✅ 定期檢查訓練指標
- ✅ 實現自動檢查點以應對 Spot 實例中斷
- ❌ 不要將 TensorBoard 綁定到 `0.0.0.0` 並公開存取
- ❌ 不要硬編碼 AWS 憑證

### 下一步

1. 設置 EC2 實例（參考 `AWS_SPOT_GPU_ANALYSIS.md`）
2. 部署 FreqAI 訓練環境
3. 使用本指南設置 TensorBoard 存取
4. 開始訓練並監控進度
5. 調整超參數以改善結果

---

## 相關資源

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [FreqAI Documentation](https://www.freqtrade.io/en/stable/freqai/)
- [AWS Systems Manager Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html)
- 本專案文檔：
  - `AWS_SPOT_GPU_ANALYSIS.md` - EC2 實例選擇和定價
  - `AWS_SPOT_FLEET_IAM_ROLE_FIX.md` - IAM 角色設置

---

**文檔版本**: 1.0  
**最後更新**: 2025年10月21日  
**適用於**: FreqAI PPO 訓練, AWS EC2, TensorBoard 監控

**作者備註**：本指南基於實際 FreqAI 訓練經驗編寫。如有任何問題或改進建議，歡迎反饋！
