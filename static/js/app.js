/**

 * 中国象棋对战机器人 - 前端控制逻辑

 */



// Socket.IO连接

const socket = io();



// 全局状态

const state = {

    cameraRunning: false,

    calibrated: false,

    robotConnected: false,

    gameStatus: 'waiting',

    currentPlayer: 'red'

};



// DOM元素

const elements = {

    videoFeed: document.getElementById('video-feed'),

    statusText: document.getElementById('status-text'),

    playerText: document.getElementById('player-text'),

    calibrationText: document.getElementById('calibration-text'),

    robotText: document.getElementById('robot-text'),

    boardState: document.getElementById('board-state'),

    moveHistory: document.getElementById('move-history'),

    logOutput: document.getElementById('log-output'),

    videoOverlay: document.getElementById('video-overlay')

};



// 鼠标悬停坐标显示
const mouseCoords = document.getElementById('mouse-coords');
elements.videoFeed.addEventListener('mousemove', (e) => {
    const rect = elements.videoFeed.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    // 显示实际像素坐标（考虑图像缩放）
    const scaleX = elements.videoFeed.naturalWidth / rect.width;
    const scaleY = elements.videoFeed.naturalHeight / rect.height;
    const actualX = Math.round(x * scaleX);
    const actualY = Math.round(y * scaleY);
    mouseCoords.textContent = `坐标: (${actualX}, ${actualY})`;
});
elements.videoFeed.addEventListener('mouseleave', () => {
    mouseCoords.textContent = '坐标: (-, -)';
});



// ==================== 工具函数 ====================



// 棋子名称转换（英文 -> 中文）

function getPieceName(class_name) {

    const pieceNames = {

        'general': '将',

        'adviser': '士',

        'elephant': '象',

        'horse': '马',

        'chariot': '车',

        'cannon': '炮',

        'soldier': '卒'

    };

    // 去掉 color 前缀，如 'red_general' -> 'general'

    const name = class_name.split('_').pop();

    return pieceNames[name] || name;

}



function log(message, type = 'info') {

    const entry = document.createElement('div');

    entry.className = `log-entry ${type}`;

    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;

    elements.logOutput.appendChild(entry);

    elements.logOutput.scrollTop = elements.logOutput.scrollHeight;

}



function updateStatus(status, player) {

    state.gameStatus = status;

    state.currentPlayer = player;

    elements.statusText.textContent = `状态: ${getStatusText(status)}`;

    elements.playerText.textContent = `当前玩家: ${player === 'red' ? '红方' : '黑方'}`;

}



function getStatusText(status) {

    const statusMap = {

        'waiting': '等待中',

        'calibrating': '标定中',

        'playing': '游戏中',

        'paused': '暂停'

    };

    return statusMap[status] || status;

}



function updateCalibrationStatus(calibrated, error = null) {

    state.calibrated = calibrated;

    if (calibrated) {

        elements.calibrationText.textContent = `已标定 (误差: ${error?.toFixed(1) || '?'}mm)`;

        elements.calibrationText.className = 'status-ok';

    } else {

        elements.calibrationText.textContent = '未标定';

        elements.calibrationText.className = 'status-error';

    }

}



function updateRobotStatus(connected) {
    state.robotConnected = connected;
    elements.robotText.textContent = connected ? '机械臂: 已连接' : '机械臂: 未连接';
    elements.robotText.className = connected ? 'status-ok' : 'status-error';

    // 更新 header 状态
    const headerStatus = document.getElementById('header-robot-status');
    if (headerStatus) {
        headerStatus.textContent = connected ? '机械臂: 已连接' : '机械臂: 未连接';
        headerStatus.className = connected ? 'header-status ok' : 'header-status';
    }
}



function showLoading(button, text = '处理中...') {

    button.dataset.originalText = button.textContent;

    button.textContent = text;

    button.disabled = true;

}



function hideLoading(button) {

    button.textContent = button.dataset.originalText;

    button.disabled = false;

}



// ==================== API调用 ====================



async function apiCall(url, method = 'GET', data = null) {

    try {

        const options = {

            method: method,

            headers: {}

        };



        if (data && method !== 'GET') {

            options.headers['Content-Type'] = 'application/json';

            options.body = JSON.stringify(data);

        }



        const response = await fetch(url, options);

        const result = await response.json();

        return result;

    } catch (error) {

        log(`API调用失败: ${error.message}`, 'error');

        return { success: false, error: error.message };

    }

}



// ==================== 摄像头控制 ====================



document.getElementById('btn-camera-start').addEventListener('click', async function() {

    showLoading(this);

    const result = await apiCall('/api/camera/start', 'POST');

    if (result.success) {

        state.cameraRunning = true;

        log('摄像头已启动', 'success');

    } else {

        log('摄像头启动失败', 'error');

    }

    hideLoading(this);

});



document.getElementById('btn-camera-stop').addEventListener('click', async function() {

    const result = await apiCall('/api/camera/stop', 'POST');

    state.cameraRunning = false;

    log('摄像头已停止');

});



document.getElementById('btn-snapshot').addEventListener('click', async function() {

    const result = await apiCall('/api/camera/snapshot', 'POST');

    if (result.success) {

        log(`已保存截图: ${result.filename}`, 'success');

    } else {

        log('截图失败', 'error');

    }

});



// ==================== 标定控制 ====================

const calibElements = {
    btnStart: document.getElementById('btn-calib-start'),
    btnPlace: document.getElementById('btn-calib-place'),
    btnReset: document.getElementById('btn-calib-reset'),
    btnScanBoard: document.getElementById('btn-scan-board-corners'),
    stepsDiv: document.getElementById('calib-steps'),
    resultDiv: document.getElementById('calib-result'),
    progress: document.getElementById('calib-progress'),
    instruction: document.getElementById('calib-instruction'),
    posInfo: document.getElementById('calib-pos-info'),
    lastPoint: document.getElementById('calib-last-point'),
    resultText: document.getElementById('calib-result-text')
};

let calibState = {
    active: false,
    currentIndex: 0
};

// 开始标定
calibElements.btnStart.addEventListener('click', async function() {
    if (!state.cameraRunning) {
        alert('请先启动摄像头！');
        return;
    }
    if (!state.robotConnected) {
        alert('请先连接机械臂！');
        return;
    }

    showLoading(this, '启动中...');
    const result = await apiCall('/api/calibration/start', 'POST');
    hideLoading(this);

    if (result.success) {
        calibState.active = true;
        calibState.currentIndex = 0;

        // 显示步骤区域
        calibElements.stepsDiv.style.display = 'block';
        calibElements.resultDiv.style.display = 'none';

        // 更新进度和提示
        updateCalibUI(result.current);

        log('棋子标定已开始');
    } else {
        log(`启动失败: ${result.error}`, 'error');
    }
});

// 放置并检测
calibElements.btnPlace.addEventListener('click', async function() {
    showLoading(this, '放置检测中...');
    const result = await apiCall('/api/calibration/place', 'POST');
    hideLoading(this);

    if (result.success) {
        if (result.completed) {
            // 标定完成
            calibElements.stepsDiv.style.display = 'none';
            calibElements.resultDiv.style.display = 'block';
            calibElements.resultText.innerHTML = `<span class="status-ok">✓ 完成</span>`;
            updateCalibrationStatus(true, result.error);
            updateCalibStatusHeader(true, result);
            log(result.message, 'success');
            calibState.active = false;
        } else {
            // 更新进度
            updateCalibUI(result.current);

            // 显示上一个点的信息
            if (result.last_point) {
                calibElements.lastPoint.innerHTML = `
                    <span class="status-ok">✓ ${result.last_point.id}</span>
                    像素: (${result.last_point.pixel[0].toFixed(0)}, ${result.last_point.pixel[1].toFixed(0)})
                `;
            }

            log(result.message, 'success');
        }
    } else {
        log(`放置失败: ${result.error}`, 'error');
    }
});

// 重新标定
calibElements.btnReset.addEventListener('click', async function() {
    const result = await apiCall('/api/calibration/reset', 'POST');

    if (result.success) {
        calibElements.stepsDiv.style.display = 'none';
        calibElements.resultDiv.style.display = 'none';
        calibElements.lastPoint.textContent = '';
        calibState.active = false;
        updateCalibrationStatus(false);
        updateCalibStatusHeader(false);
        log('标定已重置');
    }
});

// 测试抓取精度
document.getElementById('btn-test-grab').addEventListener('click', async function() {
    // 检查标定状态
    if (!state.calibrated) {
        alert('请先完成棋子标定！');
        return;
    }

    showLoading(this, '测试中...');

    const result = await apiCall('/api/calibration/test_grab', 'POST');

    if (result.success) {
        log(`测试抓取: ${result.piece} (置信度: ${result.confidence.toFixed(2)})`, 'info');
        log(`像素: (${result.pixel[0].toFixed(0)}, ${result.pixel[1].toFixed(0)}) → 机械臂: (${result.robot[0].toFixed(1)}, ${result.robot[1].toFixed(1)})`, 'info');
        log(`放置到 ${result.place_zone} 区`, 'info');
    } else {
        log(`测试失败: ${result.error}`, 'error');
        alert(result.error);
    }

    hideLoading(this);
});

// 测试抓取偏差
document.getElementById('btn-test-offset').addEventListener('click', async function() {
    if (!state.calibrated) {
        alert('请先完成棋子标定！');
        return;
    }

    showLoading(this, '检测中...');

    const result = await apiCall('/api/test/grab-offset', 'POST');

    if (result.success) {
        const s = result.summary;
        log(`偏差: ${s.count}个棋子, 平均${s.avg_distance_mm}mm, 最大${s.max_distance_mm}mm`, 'info');
        result.results.forEach(r => {
            log(`  ${r.piece}@${r.board_pos}: ${r.distance_mm}mm (px:${r.offset_px[0]},${r.offset_px[1]})`, 'info');
        });
        log(`CSV: ${result.csv}`, 'info');
        if (result.image) {
            window.open(result.image, '_blank');
        }
    } else {
        log(`偏差检测失败: ${result.error}`, 'error');
        alert(result.error);
    }

    hideLoading(this);
});

// 放置棋子到指定交叉点
document.getElementById('btn-test-place').addEventListener('click', async function() {
    if (!state.robotConnected) {
        alert('请先连接机械臂');
        return;
    }

    showLoading(this, '放置中...');

    const result = await apiCall('/api/test/place-piece', 'POST');

    if (result.success) {
        log(`已放置到 ${result.position} (${result.robot[0]}, ${result.robot[1]})`, 'info');
    } else {
        log(`放置失败: ${result.error}`, 'error');
        alert(result.error);
    }

    hideLoading(this);
});

// 重置偏差测试 CSV
document.getElementById('btn-test-reset-csv').addEventListener('click', async function() {
    const result = await apiCall('/api/test/reset-csv', 'POST');
    if (result.success) {
        log('偏差测试 CSV 已重置', 'info');
    }
});

// 扫描棋盘更新交叉点
calibElements.btnScanBoard.addEventListener('click', async function() {
    showLoading(this, '扫描中...');
    const result = await apiCall('/api/calibration/scan_board', 'POST');
    hideLoading(this);

    if (result.success) {
        log(`棋盘定位完成：${result.intersections_count} 个交叉点`, 'success');
    } else {
        log(`棋盘定位失败: ${result.error}`, 'error');
        if (result.cars_needed) {
            alert(`需要四个车在标准位置：${result.cars_needed.join(', ')}`);
        }
    }
});

// 更新标定UI
function updateCalibUI(current) {
    if (!current) return;

    calibElements.progress.textContent = `(${current.index}/${current.total})`;
    calibElements.instruction.textContent = current.instruction;
    calibElements.posInfo.textContent = `${current.name} (${current.robot[0].toFixed(1)}, ${current.robot[1].toFixed(1)})`;
}

// 更新标定状态（简化显示）
function updateCalibStatusHeader(calibrated, data = {}) {
    const headerStatus = document.getElementById('calib-header-status');
    if (calibrated) {
        headerStatus.textContent = '已标定';
        headerStatus.className = 'calib-header-status ok';
    } else {
        headerStatus.textContent = '未标定';
        headerStatus.className = 'calib-header-status';
    }
}

// 获取标定状态
async function updateCalibStatus() {
    const result = await apiCall('/api/calibration/status');
    updateCalibrationStatus(result.calibrated, result.error);
    updateCalibStatusHeader(result.calibrated, result);
}



// ==================== 机械臂控制 ====================



document.getElementById('btn-robot-connect').addEventListener('click', async function() {

    showLoading(this, '连接中...');

    const result = await apiCall('/api/robot/connect', 'POST');

    updateRobotStatus(result.connected);

    if (result.connected) {

        log('机械臂已连接', 'success');

    } else {

        log('机械臂连接失败', 'error');

    }

    hideLoading(this);

});



document.getElementById('btn-robot-disconnect').addEventListener('click', async function() {

    const result = await apiCall('/api/robot/disconnect', 'POST');

    updateRobotStatus(false);

    log('机械臂已断开');

});



document.getElementById('btn-robot-zero').addEventListener('click', async function() {

    const result = await apiCall('/api/robot/connect', 'POST');

    log('机械臂回零位');

});



// ==================== 游戏控制 ====================


document.getElementById('btn-game-start').addEventListener('click', async function() {
    if (!state.calibrated) {
        alert('请先完成相机标定！');
        return;
    }

    showLoading(this, '扫描棋盘中...');
    const result = await apiCall('/api/game/start', 'POST');

    if (result.success) {
        updateStatus('playing', 'red');
        log('游戏开始！', 'success');
        displayBoard(result.board);
    } else {
        // 显示详细错误信息
        if (result.errors && result.errors.length > 0) {
            let errorMsg = '棋盘布局错误：\n\n';
            result.errors.forEach(err => {
                errorMsg += '• ' + err + '\n';
            });
            alert(errorMsg);
        } else {
            alert(`游戏启动失败：${result.error}`);
        }
    }

    hideLoading(this);
});


// 自定义开局按钮
document.getElementById('btn-game-custom').addEventListener('click', async function() {
    if (!state.calibrated) {
        alert('请先完成相机标定！');
        return;
    }

    showLoading(this, '扫描棋盘中...');
    const result = await apiCall('/api/game/start', 'POST', { mode: 'custom' });

    if (result.success) {
        updateStatus('playing', 'red');
        log(`自定义开局成功！红方 ${Object.values(result.board.pieces).filter(p => p.color === 'red').length} 个棋子，黑方 ${Object.values(result.board.pieces).filter(p => p.color === 'black').length} 个棋子`, 'success');
        displayBoard(result.board);
    } else {
        alert(`自定义开局失败：${result.error}`);
    }

    hideLoading(this);
});




document.getElementById('btn-game-stop').addEventListener('click', async function() {

    const result = await apiCall('/api/game/stop', 'POST');

    updateStatus('waiting', state.currentPlayer);

    log('游戏已停止');

});



document.getElementById('btn-game-reset').addEventListener('click', async function() {

    const result = await apiCall('/api/game/reset', 'POST');

    updateStatus('waiting', 'red');

    elements.boardState.innerHTML = '<p>点击"扫描棋盘"查看棋子分布</p>';

    elements.moveHistory.innerHTML = '';

    // 恢复开始游戏按钮状态
    const startBtn = document.getElementById('btn-game-start');
    startBtn.textContent = '开始游戏';
    startBtn.disabled = false;

    log('游戏已重置');

});



document.getElementById('btn-scan-board').addEventListener('click', async function() {

    showLoading(this, '扫描中...');

    const result = await apiCall('/api/board/scan', 'POST');

    if (result.success) {

        log(`扫描完成，检测到 ${result.count} 个棋子`, 'success');

        displayBoard(result);

    } else {

        log(`扫描失败: ${result.error}`, 'error');

    }

    hideLoading(this);

});



// ==================== AI控制 ====================

// 难度调节
document.getElementById('ai-difficulty').addEventListener('change', async function() {
    const level = parseInt(this.value);
    const result = await apiCall('/api/game/difficulty', 'POST', { level: level });
    if (result.success) {
        log(`AI 难度设置为: ${level}`, 'info');
    } else {
        log(`设置难度失败: ${result.error}`, 'error');
    }
});

// 引擎开关
document.getElementById('ai-engine').addEventListener('change', async function() {
    const useEngine = this.checked;
    const result = await apiCall('/api/game/difficulty', 'POST', { use_engine: useEngine });
    if (result.success) {
        log(`引擎模式: ${useEngine ? '开启' : '关闭'}`, 'info');
    } else {
        log(`设置引擎模式失败: ${result.error}`, 'error');
    }
});



document.getElementById('btn-ai-move').addEventListener('click', async function() {

    if (state.gameStatus !== 'playing') {

        alert('请先开始游戏！');

        return;

    }



    if (!state.robotConnected) {

        alert('请先连接机械臂！');

        return;

    }



    showLoading(this, '检测并思考中...');



    const result = await apiCall('/api/game/move/ai', 'POST');



    if (result.success) {

        // 显示人类走棋信息

        if (result.human_move) {

            const humanMove = result.human_move;

            const pieceName = humanMove.moving_piece ? getPieceName(humanMove.moving_piece.class_name) : '棋子';

            let humanMsg = `红方 ${pieceName} 从 ${humanMove.from_pos} 移动到 ${humanMove.to_pos}`;

            if (humanMove.captured) {

                const capturedName = getPieceName(humanMove.captured.class_name);

                humanMsg += ` - 吃黑${capturedName}`;

            }

            log(humanMsg, 'info');

            addMoveToHistory('红方', humanMove.from_pos, humanMove.to_pos, pieceName, humanMove.captured);

        }



        // 显示AI走棋信息

        if (result.ai_move) {

            const aiMove = result.ai_move;

            log(`AI计算走棋: ${aiMove.from} -> ${aiMove.to}`, 'info');

        }



        log('机械臂正在执行...', 'info');

    } else {

        log(`操作失败: ${result.error}`, 'error');

        alert(result.error);

    }



    hideLoading(this);

});



// ==================== 显示函数 ====================



function displayBoard(data) {

    const pieces = data.pieces || {};

    const count = Object.keys(pieces).length;



    let html = `<p>检测到 ${count} 个棋子</p>`;



    if (count > 0) {

        html += '<div style="margin-top: 10px;">';

        html += '<table style="width: 100%; border-collapse: collapse;">';

        html += '<tr style="border-bottom: 1px solid #0f3460;">';

        html += '<th style="text-align: left; padding: 5px;">位置</th>';

        html += '<th style="text-align: left; padding: 5px;">棋子</th>';

        html += '<th style="text-align: left; padding: 5px;">颜色</th>';

        html += '</tr>';



        for (const [pos, piece] of Object.entries(pieces)) {

            const colorClass = piece.color === 'red' ? 'status-error' : 'status-ok';

            html += `<tr style="border-bottom: 1px solid rgba(15, 52, 96, 0.3);">`;

            html += `<td style="padding: 5px;">${pos}</td>`;

            html += `<td style="padding: 5px;">${piece.class_name}</td>`;

            html += `<td style="padding: 5px;" class="${colorClass}">${piece.color === 'red' ? '红' : '黑'}</td>`;

            html += '</tr>';

        }



        html += '</table></div>';

    }



    elements.boardState.innerHTML = html;

}



function addMoveToHistory(player, from, to, pieceName = '', captured = null, isAI = false) {
    const item = document.createElement('div');
    item.className = 'history-item';
    const playerText = player === 'red' ? '红方' : '黑方';
    const aiBadge = isAI ? ' [AI]' : '';
    
    let moveText = `${from} → ${to}`;
    if (pieceName) {
        moveText = `${pieceName}${moveText}`;
    }
    
    // 如果有吃子，添加吃子信息
    if (captured) {
        const capturedName = getPieceName(captured.class_name);
        const capturedColor = captured.color === 'red' ? '红' : '黑';
        moveText += ` 吃${capturedColor}${capturedName}`;
    }
    
    item.innerHTML = `<span class="${player === 'red' ? 'status-error' : 'status-ok'}">${playerText}${aiBadge}</span>: ${moveText}`;
    elements.moveHistory.insertBefore(item, elements.moveHistory.firstChild);
}





// ==================== Socket.IO事件 ====================



socket.on('connect', function() {

    log('已连接到服务器', 'success');

});



socket.on('disconnect', function() {

    log('与服务器断开连接', 'warning');

});



socket.on('status', function(data) {

    log(data.message);

});



// move_detected 事件已移除 - 改为在点击AI走棋时主动检测



socket.on('player_changed', function(data) {

    state.currentPlayer = data.current_player;



    const playerText = data.current_player === 'red' ? '红方' : '黑方';



    elements.playerText.textContent = `当前玩家：${playerText}`;



    log(`切换到${playerText}`, 'info');



    // 如果有新的棋盘状态，更新显示

    if (data.board) {

        displayBoard(data.board);

    }



    // 如果有AI走棋信息，添加到历史

    if (data.ai_move) {

        const aiMove = data.ai_move;

        const pieceName = aiMove.piece ? getPieceName(aiMove.piece) : '棋子';

        log(`AI走棋完成: ${aiMove.from} -> ${aiMove.to}`, 'success');

        addMoveToHistory('黑方', aiMove.from, aiMove.to, pieceName, null, true);

    }

});



socket.on('ai_move_failed', function(data) {

    log(`AI走棋失败: ${data.error}`, 'error');

    alert(`AI走棋失败: ${data.error}`);

});



socket.on('move_invalid', function(data) {

    log(`非法走棋：${data.from} -> ${data.to} - ${data.reason}`, 'error');

    alert(`非法走法：${data.from} -> ${data.to}

${data.reason}`);

});


// ==================== 语音交互 ====================

const voiceElements = {
    panel: document.getElementById('voice-panel'),
    statusText: document.getElementById('voice-status-text'),
    speakingIndicator: document.getElementById('voice-speaking-indicator'),
    listeningIndicator: document.getElementById('voice-listening-indicator'),
    commentaryDisplay: document.getElementById('commentary-display'),
    btnInterrupt: document.getElementById('btn-voice-interrupt'),
    autoCommentary: document.getElementById('voice-auto-commentary'),
    characterSelector: document.getElementById('character-selector'),
    commentaryCharacterSelector: document.getElementById('commentary-character-selector'),
    // 会话模式元素
    commentaryModeContent: document.getElementById('commentary-mode-content'),
    sessionModeContent: document.getElementById('session-mode-content'),
    dialogueHistory: document.getElementById('dialogue-history'),
    dialogueMessages: document.querySelector('.dialogue-messages'),
    btnSessionStart: document.getElementById('btn-session-start'),
    btnSessionStop: document.getElementById('btn-session-stop')
};

const voiceState = {
    enabled: true,
    autoCommentary: true,
    mode: 'commentary',  // 'commentary' or 'session'
    sessionActive: false,
    currentCharacter: null,
    isListening: false
};

// 初始化语音设置
async function initVoiceSettings() {
    const result = await apiCall('/api/voice/status');
    if (result.enabled !== undefined) {
        voiceState.enabled = result.enabled;
        // 如果语音模块不可用，隐藏面板
        if (!result.enabled || !result.available) {
            if (voiceElements.panel) {
                voiceElements.panel.style.display = 'none';
            }
        }
    }

    const settings = await apiCall('/api/voice/settings');
    if (settings.auto_commentary !== undefined) {
        voiceState.autoCommentary = settings.auto_commentary;
        if (voiceElements.autoCommentary) {
            voiceElements.autoCommentary.checked = settings.auto_commentary;
        }
    }

    // 初始化角色选择
    await initCharacterSelector();

    // 初始化解说角色选择
    await initCommentaryCharacterSelector();
}

// 初始化解说角色选择器
async function initCommentaryCharacterSelector() {
    const result = await apiCall('/api/voice/commentary-character/list');
    if (result.characters && voiceElements.commentaryCharacterSelector) {
        voiceElements.commentaryCharacterSelector.innerHTML = '';
        result.characters.forEach(char => {
            const option = document.createElement('option');
            option.value = char.id;
            option.textContent = char.name;
            voiceElements.commentaryCharacterSelector.appendChild(option);
        });

        // 获取当前解说角色
        const current = await apiCall('/api/voice/commentary-character/current');
        if (current.character && voiceElements.commentaryCharacterSelector) {
            voiceElements.commentaryCharacterSelector.value = current.character.id;
        }
    }
}

// 初始化角色选择器
async function initCharacterSelector() {
    const result = await apiCall('/api/voice/character/list');
    if (result.characters && voiceElements.characterSelector) {
        // 更新下拉选项
        voiceElements.characterSelector.innerHTML = '';
        result.characters.forEach(char => {
            const option = document.createElement('option');
            option.value = char.id;
            option.textContent = char.name;
            if (char.is_current) {
                option.selected = true;
                voiceState.currentCharacter = char;
            }
            voiceElements.characterSelector.appendChild(option);
        });
    }
}

// 打断播报
if (voiceElements.btnInterrupt) {
    voiceElements.btnInterrupt.addEventListener('click', async function() {
        const result = await apiCall('/api/voice/interrupt', 'POST');
        if (result.success) {
            log('已打断播报', 'info');
            voiceElements.speakingIndicator.style.display = 'none';
            voiceElements.statusText.textContent = '状态: 待命';
        }
    });
}

// 自动解说开关
if (voiceElements.autoCommentary) {
    voiceElements.autoCommentary.addEventListener('change', async function() {
        const result = await apiCall('/api/voice/settings', 'POST', {
            auto_commentary: this.checked
        });
        voiceState.autoCommentary = this.checked;
        log(`自动解说: ${this.checked ? '开启' : '关闭'}`, 'info');
    });
}

// ========== 角色选择 ==========

if (voiceElements.characterSelector) {
    voiceElements.characterSelector.addEventListener('change', async function() {
        const characterId = this.value;
        const result = await apiCall('/api/voice/character/set', 'POST', {
            character_id: characterId
        });

        if (result.success) {
            voiceState.currentCharacter = result.character;
            log(`角色切换为: ${result.character.name}`, 'info');
        }
    });
}

// 解说角色选择
if (voiceElements.commentaryCharacterSelector) {
    voiceElements.commentaryCharacterSelector.addEventListener('change', async function() {
        const characterId = this.value;
        const result = await apiCall('/api/voice/commentary-character/set', 'POST', {
            character_id: characterId
        });

        if (result.success) {
            log(`解说风格切换为: ${result.character.name}`, 'info');
        }
    });
}

// ========== 模式切换 ==========

document.querySelectorAll('input[name="voice-mode"]').forEach(radio => {
    radio.addEventListener('change', async function() {
        const mode = this.value;

        // 切换 UI
        if (mode === 'commentary') {
            voiceElements.commentaryModeContent.style.display = 'block';
            voiceElements.sessionModeContent.style.display = 'none';
        } else {
            voiceElements.commentaryModeContent.style.display = 'none';
            voiceElements.sessionModeContent.style.display = 'block';
        }

        // 通知后端
        const result = await apiCall('/api/voice/mode', 'POST', { mode: mode });
        if (result.success) {
            voiceState.mode = mode;
            log(`切换到${mode === 'commentary' ? '解说' : '会话'}模式`, 'info');
        }
    });
});

// ========== 解说模式持续监听控制 ==========

// 开始解说监听
document.getElementById('btn-commentary-listening-start').addEventListener('click', async function() {
    log('正在启动解说监听...', 'info');

    const result = await apiCall('/api/voice/commentary/listening/start', 'POST');

    if (result.success) {
        log('解说监听已启动，请走棋后说"下好了"', 'success');
    } else {
        log(`启动解说监听失败: ${result.error}`, 'error');
    }
});

// 停止解说监听
document.getElementById('btn-commentary-listening-stop').addEventListener('click', async function() {
    const result = await apiCall('/api/voice/commentary/listening/stop', 'POST');

    if (result.success) {
        document.getElementById('btn-commentary-listening-start').disabled = false;
        document.getElementById('btn-commentary-listening-stop').disabled = true;
        const statusEl = document.getElementById('commentary-listening-status');
        statusEl.textContent = '未激活';
        statusEl.classList.remove('status-active');
        statusEl.classList.add('status-idle');
        log('解说监听已停止', 'info');
    }
});

// ========== 会话控制 ==========

// 开始会话
if (voiceElements.btnSessionStart) {
    voiceElements.btnSessionStart.addEventListener('click', async function() {
        log('正在启动会话...', 'info');

        const result = await apiCall('/api/voice/session/start', 'POST');

        if (result.success) {
            voiceState.sessionActive = true;

            // 更新 UI
            voiceElements.btnSessionStart.style.display = 'none';
            voiceElements.btnSessionStop.style.display = 'inline-block';
            voiceElements.listeningIndicator.style.display = 'inline';

            // 清空对话历史
            if (voiceElements.dialogueMessages) {
                voiceElements.dialogueMessages.innerHTML = '';
            }

            // 显示问候语
            if (result.greeting) {
                addDialogueMessage('assistant', result.greeting, result.character?.name || 'AI');
            }

            log('会话模式已启动', 'success');
        } else {
            log(`启动会话失败: ${result.error}`, 'error');
        }
    });
}

// 结束会话
if (voiceElements.btnSessionStop) {
    voiceElements.btnSessionStop.addEventListener('click', async function() {
        const result = await apiCall('/api/voice/session/stop', 'POST');

        if (result.success) {
            voiceState.sessionActive = false;

            // 更新 UI
            voiceElements.btnSessionStart.style.display = 'inline-block';
            voiceElements.btnSessionStop.style.display = 'none';
            voiceElements.listeningIndicator.style.display = 'none';
            voiceElements.statusText.textContent = '待命';

            log('会话已结束', 'info');
        }
    });
}

// 添加对话消息
function addDialogueMessage(role, content, sender) {
    if (!voiceElements.dialogueMessages) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = `dialogue-msg ${role}`;

    const time = new Date().toLocaleTimeString();

    msgDiv.innerHTML = `
        <span class="msg-sender">${sender}</span>
        <span class="msg-content">${content}</span>
        <span class="msg-time">${time}</span>
    `;

    voiceElements.dialogueMessages.appendChild(msgDiv);

    // 滚动到底部
    if (voiceElements.dialogueHistory) {
        voiceElements.dialogueHistory.scrollTop = voiceElements.dialogueHistory.scrollHeight;
    }
}

// ========== 旧对话控制（兼容） ==========

// 开始对话
if (voiceElements.btnDialogueStart) {
    voiceElements.btnDialogueStart.addEventListener('click', async function() {
        log('正在启动对话...', 'info');

        const result = await apiCall('/api/voice/dialogue/start', 'POST');

        if (result.success) {
            voiceState.dialogueActive = true;

            // 显示对话界面
            voiceElements.dialogueHistory.style.display = 'block';
            voiceElements.btnDialogueStop.style.display = 'inline-block';
            voiceElements.btnDialogueStart.style.display = 'none';

            // 清空对话历史显示
            if (voiceElements.dialogueMessages) {
                voiceElements.dialogueMessages.innerHTML = '';
            }

            // 显示问候语
            if (result.greeting) {
                addDialogueMessage('assistant', result.greeting, result.character?.name || 'AI');
            }

            log('对话模式已启动', 'success');
        } else {
            log(`启动对话失败: ${result.error}`, 'error');
        }
    });
}

// 结束对话
if (voiceElements.btnDialogueStop) {
    voiceElements.btnDialogueStop.addEventListener('click', async function() {
        const result = await apiCall('/api/voice/dialogue/stop', 'POST');

        if (result.success) {
            voiceState.dialogueActive = false;

            // 隐藏对话界面
            voiceElements.dialogueHistory.style.display = 'none';
            voiceElements.btnDialogueStop.style.display = 'none';
            voiceElements.btnDialogueStart.style.display = 'inline-block';

            log('对话已结束', 'info');
        }
    });
}

// Socket.IO 语音事件
socket.on('commentary_generated', function(data) {
    // 显示解说文本
    const text = data.text;
    const humanMove = data.human_move;
    const aiMove = data.ai_move;

    let html = `<div class="commentary-text">${text}</div>`;

    // 添加走棋信息元数据
    if (humanMove || aiMove) {
        html += `<div class="commentary-meta">`;
        if (humanMove) {
            html += `红方: ${humanMove.from_pos} → ${humanMove.to_pos}`;
        }
        if (aiMove) {
            html += ` | AI: ${aiMove.from} → ${aiMove.to}`;
        }
        html += `</div>`;
    }

    voiceElements.commentaryDisplay.innerHTML = html;
    voiceElements.speakingIndicator.style.display = 'inline';
    voiceElements.statusText.textContent = '状态: 播报中';

    log('解说已生成', 'success');
});

socket.on('voice_error', function(data) {
    log(`语音错误: ${data.error}`, 'error');
    voiceElements.speakingIndicator.style.display = 'none';
    voiceElements.statusText.textContent = '状态: 错误';
    setTimeout(() => {
        voiceElements.statusText.textContent = '状态: 待命';
    }, 2000);
});

// ========== 对话 Socket 事件 ==========

socket.on('dialogue_start', function(data) {
    if (data.success) {
        voiceState.dialogueActive = true;
        voiceElements.statusText.textContent = '状态: 对话中';

        // 显示问候语
        if (data.greeting) {
            addDialogueMessage('assistant', data.greeting, data.character?.name || 'AI');
        }
    }
});

socket.on('dialogue_message', function(data) {
    // 添加对话消息
    addDialogueMessage('user', data.user, '你');
    addDialogueMessage('assistant', data.assistant, data.character);
    log('收到对话回复', 'success');
});

socket.on('dialogue_stop', function(data) {
    voiceState.dialogueActive = false;
    voiceElements.statusText.textContent = '状态: 待命';
});

socket.on('listening_status', function(data) {
    if (data.status === 'listening') {
        voiceState.isListening = true;
        voiceElements.listeningIndicator.style.display = 'inline';
        voiceElements.statusText.textContent = '状态: 聆听中...';
    } else {
        voiceState.isListening = false;
        voiceElements.listeningIndicator.style.display = 'none';
        voiceElements.statusText.textContent = '状态: 处理中...';
    }
});

socket.on('character_changed', function(data) {
    if (data.character) {
        voiceState.currentCharacter = data.character;
        log(`角色已切换: ${data.character.name}`, 'info');
    }
});

// 走棋意图检测（会话模式）
socket.on('move_intent_detected', function(data) {
    log('检测到走棋意图: ' + data.transcript, 'info');

    // 调用 AI 走棋 API（包含人类走棋检测）
    apiCall('/api/game/move/ai', 'POST').then(result => {
        if (!result.success) {
            log('走棋失败: ' + result.error, 'error');
        }
    });
});

// 解说模式：走棋意图检测
socket.on('commentary_move_intent', function(data) {
    log('解说监听：检测到走棋意图 ' + data.transcript, 'info');

    // 调用 AI 走棋 API
    apiCall('/api/game/move/ai', 'POST').then(result => {
        if (!result.success) {
            log('走棋失败: ' + result.error, 'error');
        }
        // 结果会通过 commentary_move_result 事件返回到语音模块
    });
});

// 解说模式：监听启动
socket.on('commentary_listening_started', function(data) {
    if (data.success) {
        document.getElementById('btn-commentary-listening-start').disabled = true;
        document.getElementById('btn-commentary-listening-stop').disabled = false;
        const statusEl = document.getElementById('commentary-listening-status');
        statusEl.textContent = '监听中';
        statusEl.classList.remove('status-idle');
        statusEl.classList.add('status-active');
        log('解说监听已启动，请走棋后说"下好了"', 'info');
    }
});

// 解说模式：监听停止
socket.on('commentary_listening_stopped', function(data) {
    document.getElementById('btn-commentary-listening-start').disabled = false;
    document.getElementById('btn-commentary-listening-stop').disabled = true;
    const statusEl = document.getElementById('commentary-listening-status');
    statusEl.textContent = '未激活';
    statusEl.classList.remove('status-active');
    statusEl.classList.add('status-idle');
    log('解说监听已停止', 'info');
});

// 游戏结束
socket.on('game_over', function(data) {
    const winnerText = data.winner === 'red' ? '红方' : '黑方';
    log(`游戏结束！${winnerText}获胜！原因：${data.reason}`, 'info');
    alert(`游戏结束！${winnerText}获胜！\n原因：${data.reason}`);
});



// ==================== 日志控制 ====================



document.getElementById('btn-clear-log').addEventListener('click', function() {

    elements.logOutput.innerHTML = '';

});



// ==================== 初始化 ====================



async function init() {

    log('系统初始化...');



    // 获取初始状态

    const calibStatus = await apiCall('/api/calibration/status');

    updateCalibrationStatus(calibStatus.calibrated, calibStatus.error);

    // 如果已标定，显示简洁状态
    if (calibStatus.calibrated) {
        calibElements.resultDiv.style.display = 'block';
        calibElements.resultText.innerHTML = `<span class="status-ok">✓ 已标定</span>`;
        updateCalibStatusHeader(true, calibStatus);
    }


    const robotStatus = await apiCall('/api/robot/status');

    updateRobotStatus(robotStatus.connected);


    // 获取 AI 难度设置
    const difficultyStatus = await apiCall('/api/game/difficulty');
    if (difficultyStatus.difficulty) {
        document.getElementById('ai-difficulty').value = difficultyStatus.difficulty;
    }
    if (difficultyStatus.use_engine !== undefined) {
        document.getElementById('ai-engine').checked = difficultyStatus.use_engine;
    }


    const gameStatus = await apiCall('/api/game/state');

    if (gameStatus.status) {

        updateStatus(gameStatus.status, gameStatus.current_player);

        // 显示历史走棋

        if (gameStatus.move_history) {

            gameStatus.move_history.forEach(move => {

                addMoveToHistory(move.player, move.from, move.to, move.piece || '', move.captured || null, move.ai);

            });

        }

    }

    // 获取细修正标定状态
    await updateCalibStatus();

    // 初始化语音设置
    await initVoiceSettings();

    // 自动启动摄像头和连接机械臂
    await autoStartDevices();

    log('初始化完成');

}

// 自动启动设备
async function autoStartDevices() {
    // 启动摄像头
    log('自动启动摄像头...');
    const cameraResult = await apiCall('/api/camera/start', 'POST');
    if (cameraResult.success) {
        state.cameraRunning = true;
        log('摄像头已启动', 'success');
    } else {
        log('摄像头启动失败', 'error');
    }

    // 连接机械臂
    log('自动连接机械臂...');
    const robotResult = await apiCall('/api/robot/connect', 'POST');
    updateRobotStatus(robotResult.connected);
    if (robotResult.connected) {
        log('机械臂已连接', 'success');
    } else {
        log('机械臂连接失败', 'error');
    }
}



// 页面加载完成后初始化

document.addEventListener('DOMContentLoaded', init);

// 折叠面板控制（需在 DOM 加载后执行）
document.addEventListener('DOMContentLoaded', function() {
    const calibrationToggle = document.getElementById('calibration-toggle');
    if (calibrationToggle) {
        calibrationToggle.addEventListener('click', function() {
            const panel = this.parentElement;
            const content = document.getElementById('calibration-content');
            const isOpen = panel.classList.contains('open');

            if (isOpen) {
                panel.classList.remove('open');
                content.style.display = 'none';
            } else {
                panel.classList.add('open');
                content.style.display = 'block';
            }
        });
    }
});



// 错误处理

window.onerror = function(message, source, lineno, colno, error) {

    log(`JavaScript错误: ${message}`, 'error');

};

