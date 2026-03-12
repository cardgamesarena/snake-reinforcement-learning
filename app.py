import eventlet
eventlet.monkey_patch()

import os
import threading
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

from game import SnakeGame
from agent import DQNAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# --- Global training state ---
game = SnakeGame()
agent = DQNAgent()

training_active = threading.Event()   # set = running, clear = paused
training_reset  = threading.Event()   # set = stop and reinit

emit_every = 3   # emit game state every N episodes
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background training loop
# ---------------------------------------------------------------------------

def training_loop():
    global game, agent, emit_every

    episode = 0

    while not training_reset.is_set():
        training_active.wait()  # blocks (yields) when paused

        if training_reset.is_set():
            break

        state = game.reset()
        done = False
        episode_loss_sum = 0.0
        episode_steps = 0

        while not done and not training_reset.is_set():
            training_active.wait()

            action = agent.select_action(state)
            next_state, reward, done = game.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.train_step()

            if loss is not None:
                episode_loss_sum += loss
                episode_steps += 1

            state = next_state
            eventlet.sleep(0)  # yield to eventlet scheduler

        if training_reset.is_set():
            break

        score = game.get_score()
        agent.record_score(score)
        agent.decay_epsilon()
        episode += 1

        if episode % emit_every == 0:
            avg_loss = episode_loss_sum / max(1, episode_steps)
            with _lock:
                grid_data = game.get_grid_dict()

            socketio.emit('game_state', {
                "grid":  grid_data["grid"],
                "score": score,
                "steps": grid_data["steps"],
            })
            socketio.emit('metrics', {
                "episode":      episode,
                "score":        score,
                "epsilon":      round(agent.epsilon, 4),
                "loss":         round(avg_loss, 6),
                "mean_score":   round(agent.mean_score(), 2),
                "score_history": list(agent.score_history),
            })

        eventlet.sleep(0)

    socketio.emit('training_status', {"status": "reset"})


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    with _lock:
        return jsonify({
            "running": training_active.is_set(),
            **agent.get_metrics(),
        })


# ---------------------------------------------------------------------------
# SocketIO event handlers
# ---------------------------------------------------------------------------

@socketio.on('connect')
def handle_connect():
    with _lock:
        grid_data = game.get_grid_dict()
    socketio.emit('game_state', {
        "grid":  grid_data["grid"],
        "score": grid_data["score"],
        "steps": grid_data["steps"],
    })
    socketio.emit('metrics', {
        "episode":      agent.episode_count,
        "score":        game.get_score(),
        "epsilon":      round(agent.epsilon, 4),
        "loss":         0,
        "mean_score":   round(agent.mean_score(), 2),
        "score_history": list(agent.score_history),
    })
    socketio.emit('training_status', {
        "status": "running" if training_active.is_set() else "paused"
    })


@socketio.on('start_training')
def handle_start(data=None):
    global game, agent
    training_reset.clear()

    # Start thread only if not already running
    if not training_active.is_set():
        training_active.set()
        socketio.start_background_task(training_loop)

    socketio.emit('training_status', {"status": "running"})


@socketio.on('pause_training')
def handle_pause(data=None):
    training_active.clear()
    socketio.emit('training_status', {"status": "paused"})


@socketio.on('reset_training')
def handle_reset(data=None):
    global game, agent
    training_active.clear()
    training_reset.set()
    eventlet.sleep(0.1)

    with _lock:
        game = SnakeGame()
        agent = DQNAgent()

    training_reset.clear()

    with _lock:
        grid_data = game.get_grid_dict()

    socketio.emit('game_state', {
        "grid":  grid_data["grid"],
        "score": 0,
        "steps": 0,
    })
    socketio.emit('metrics', {
        "episode":      0,
        "score":        0,
        "epsilon":      1.0,
        "loss":         0,
        "mean_score":   0,
        "score_history": [],
    })
    socketio.emit('training_status', {"status": "reset"})


@socketio.on('set_speed')
def handle_speed(data):
    global emit_every
    val = int(data.get('emit_every', 3))
    emit_every = max(1, min(50, val))


@socketio.on('ping_keepalive')
def handle_ping(data=None):
    pass  # just keep the connection alive


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
