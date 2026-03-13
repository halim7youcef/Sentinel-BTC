/**
 * app.js  —  BTC ML Trading Sentinel frontend logic
 * ================================================
 * - WebSocket live price stream from /ws/live
 * - Auto-scheduled inference every 5 minutes (synced to 5m boundaries)
 * - Lightweight-Charts candlestick chart
 * - Model breakdown bars
 * - Trade history table
 */

const API = window.location.origin;
const WS_URL = (window.location.protocol === 'https:' ? 'wss' : 'ws') + `://${window.location.host}/ws/live`;

// ── State ─────────────────────────────────────────────────────
let chart = null;
let candleSeries = null;
let lastPrice = null;
let activeLimit = 50;
let inferRunning = false;
let wsConn = null;
let priceTicker = null;    // fallback polling

// ── DOM refs ──────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const headerPrice = $('headerPrice');
const wsLabel = $('wsLabel');
const wsDot = $('wsDot');
const kpiPriceVal = $('kpiPriceVal');
const kpiPriceChange = $('kpiPriceChange');
const kpiSignalVal = $('kpiSignalVal');
const kpiSignalConf = $('kpiSignalConf');
const kpiProbVal = $('kpiProbVal');
const kpiBalanceVal = $('kpiBalanceVal');
const kpiPnl = $('kpiPnl');
const signalBadge = $('signalBadge');
const signalTime = $('signalTime');
const signalEntryPrice = $('signalEntryPrice');
const signalStake = $('signalStake');
const gaugeFill = $('gaugeFill');
const gaugeThumb = $('gaugeThumb');
const gaugeCenterVal = $('gaugeCenterVal');
const inferBtn = $('inferBtn');
const inferStatus = $('inferStatus');
const historyBody = $('historyBody');
const toast = $('toast');
const pricePulse = $('pricePulse');

// ═══════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════
function fmtPrice(v) {
    return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function fmtPct(v) {
    const s = (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
    return s;
}
function fmtTS(ts) {
    const d = new Date(ts.endsWith('Z') ? ts : ts + 'Z');
    return d.toUTCString().replace(' GMT', '');
}
function fmtShortTS(ts) {
    const d = new Date(ts.endsWith('Z') ? ts : ts + 'Z');
    return d.toISOString().substring(11, 19);
}

function flashPricePulse() {
    pricePulse.classList.remove('flash');
    void pricePulse.offsetWidth;
    pricePulse.classList.add('flash');
}

function showToast(msg, type = '') {
    toast.textContent = msg;
    toast.className = 'toast show ' + type;
    setTimeout(() => { toast.className = 'toast'; }, 3000);
}

// ═══════════════════════════════════════════════════════════════
// Price update
// ═══════════════════════════════════════════════════════════════
function updatePrice(price) {
    const priceNum = parseFloat(price);
    const prev = lastPrice;
    lastPrice = priceNum;

    const formatted = fmtPrice(priceNum);
    headerPrice.textContent = formatted;
    kpiPriceVal.textContent = formatted;
    flashPricePulse();

    if (prev !== null) {
        const delta = priceNum - prev;
        const pct = delta / prev;
        const dir = delta >= 0 ? 'up' : 'down';
        headerPrice.className = 'header-price ' + dir;
        kpiPriceChange.textContent = `${fmtPct(pct)} vs last tick`;
        kpiPriceChange.style.color = delta >= 0 ? 'var(--green)' : 'var(--red)';
        setTimeout(() => { headerPrice.className = 'header-price'; }, 600);
    } else {
        kpiPriceChange.textContent = 'Live from Binance';
    }
}

// ═══════════════════════════════════════════════════════════════
// WebSocket
// ═══════════════════════════════════════════════════════════════
function connectWS() {
    wsConn = new WebSocket(WS_URL);

    wsConn.onopen = () => {
        wsDot.className = 'ws-dot connected';
        wsLabel.textContent = 'Live';
        stopPricePolling();
    };

    wsConn.onmessage = evt => {
        try {
            const data = JSON.parse(evt.data);
            if (data.type === 'price' && data.price) updatePrice(data.price);
        } catch { }
    };

    wsConn.onerror = () => {
        wsDot.className = 'ws-dot error';
        wsLabel.textContent = 'WS Error';
    };

    wsConn.onclose = () => {
        wsDot.className = 'ws-dot';
        wsLabel.textContent = 'Reconnecting…';
        startPricePolling();
        setTimeout(connectWS, 5000);
    };
}

// Fallback REST price polling
function startPricePolling() {
    if (priceTicker) return;
    priceTicker = setInterval(async () => {
        try {
            const r = await fetch(`${API}/api/price`);
            const d = await r.json();
            if (d.price) updatePrice(d.price);
        } catch { }
    }, 5000);
}
function stopPricePolling() {
    if (priceTicker) { clearInterval(priceTicker); priceTicker = null; }
}

// ═══════════════════════════════════════════════════════════════
// Chart
// ═══════════════════════════════════════════════════════════════
function initChart() {
    const container = $('chartContainer');
    chart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: 'transparent' },
            textColor: '#7a8ba8',
        },
        grid: {
            vertLines: { color: 'rgba(255,255,255,0.04)' },
            horzLines: { color: 'rgba(255,255,255,0.04)' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.08)' },
        timeScale: {
            borderColor: 'rgba(255,255,255,0.08)',
            timeVisible: true, secondsVisible: false,
        },
        handleScroll: true, handleScale: true,
        width: container.clientWidth,
        height: 360,
    });

    candleSeries = chart.addCandlestickSeries({
        upColor: '#30d991',
        downColor: '#ff4e6a',
        borderUpColor: '#30d991',
        borderDownColor: '#ff4e6a',
        wickUpColor: '#30d991',
        wickDownColor: '#ff4e6a',
    });

    window.addEventListener('resize', () => {
        chart.applyOptions({ width: container.clientWidth });
    });
}

async function loadCandles(limit = 50) {
    try {
        const r = await fetch(`${API}/api/candles?limit=${limit}`);
        const d = await r.json();
        if (!d.candles || d.error) { showToast('Chart error: ' + (d.error || 'unknown'), 'error'); return; }

        const mapped = d.candles.map(c => ({
            time: Math.floor(new Date(c.timestamp).getTime() / 1000),
            open: c.open, high: c.high, low: c.low, close: c.close,
        }));
        mapped.sort((a, b) => a.time - b.time);
        candleSeries.setData(mapped);
        chart.timeScale().fitContent();
    } catch (e) {
        showToast('Chart fetch failed', 'error');
    }
}

// Chart limit buttons
document.querySelectorAll('.chart-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeLimit = parseInt(btn.dataset.limit);
        loadCandles(activeLimit);
    });
});
$('refreshChart').addEventListener('click', () => loadCandles(activeLimit));

// ═══════════════════════════════════════════════════════════════
// Signal + Model UI
// ═══════════════════════════════════════════════════════════════
function setGauge(prob) {
    const pct = Math.round(prob * 100);
    const fillWidth = `${prob * 100}%`;
    const thumbLeft = `${prob * 100}%`;

    gaugeFill.style.width = fillWidth;
    gaugeThumb.style.left = thumbLeft;
    gaugeCenterVal.textContent = pct + '%';

    // Colour fill based on side
    if (prob > 0.5) {
        gaugeFill.style.background = `linear-gradient(to right, var(--red), var(--gold), var(--green))`;
    } else {
        gaugeFill.style.background = `linear-gradient(to right, var(--red), var(--gold))`;
    }
}

function setModelBar(id, probRaw) {
    const barEl = $('bar-' + id);
    const probEl = $('prob-' + id);
    if (!barEl || !probEl) return;
    barEl.style.width = `${probRaw * 100}%`;
    probEl.textContent = (probRaw * 100).toFixed(1) + '%';
}

function renderSignal(data) {
    const sig = data.signal || '—';
    const prob = data.ensemble_prob ?? 0.5;
    const conf = data.confidence ?? 0;
    const price = data.btc_price ?? 0;
    const stake = data.stake ?? 0;
    const pnl = data.pnl ?? 0;
    const bal = data.virtual_balance ?? 0;

    // KPI updates
    kpiSignalVal.textContent = sig;
    kpiSignalVal.dataset.signal = sig;
    kpiSignalConf.textContent = `Confidence: ${conf.toFixed(1)}%`;
    kpiProbVal.textContent = (prob * 100).toFixed(2) + '%';
    kpiBalanceVal.textContent = fmtPrice(bal);

    const pnlSign = pnl >= 0 ? '+' : '';
    kpiPnl.textContent = `Last PnL: ${pnlSign}$${Math.abs(pnl).toFixed(2)}`;
    kpiPnl.style.color = pnl > 0 ? 'var(--green)' : pnl < 0 ? 'var(--red)' : 'var(--text-muted)';

    // Signal card
    signalBadge.textContent = sig;
    signalBadge.dataset.signal = sig;
    signalTime.textContent = data.timestamp ? fmtShortTS(data.timestamp) + ' UTC' : '—';
    signalEntryPrice.textContent = fmtPrice(price);
    signalStake.textContent = stake > 0 ? `$${stake.toFixed(2)}` : '—';

    // Gauge
    setGauge(prob);

    // Model bars
    setModelBar('xgb', data.xgb_prob ?? 0.5);
    setModelBar('lgb', data.lgb_prob ?? 0.5);
    setModelBar('cb', data.cb_prob ?? 0.5);
    setModelBar('rf', data.rf_prob ?? 0.5);
    setModelBar('ens', prob);
}

// ═══════════════════════════════════════════════════════════════
// Inference
// ═══════════════════════════════════════════════════════════════
async function runInference(auto = false) {
    if (inferRunning) return;
    inferRunning = true;
    inferBtn.disabled = true;

    inferStatus.className = 'infer-status loading';
    inferStatus.textContent = auto ? '⏱ Auto-inference running…' : '⏳ Running ML ensemble…';

    try {
        const r = await fetch(`${API}/api/signal`);
        const data = await r.json();

        if (data.error) throw new Error(data.error);

        renderSignal(data);
        loadHistory();
        // also refresh chart
        loadCandles(activeLimit);

        inferStatus.className = 'infer-status done';
        inferStatus.textContent = '✓ Inference complete';
        showToast(`Signal: ${data.signal} | P(UP): ${(data.ensemble_prob * 100).toFixed(1)}%`, 'success');
    } catch (e) {
        inferStatus.className = 'infer-status error';
        inferStatus.textContent = '✗ ' + e.message;
        showToast('Inference failed: ' + e.message, 'error');
    } finally {
        inferRunning = false;
        inferBtn.disabled = false;
        setTimeout(() => { inferStatus.textContent = ''; inferStatus.className = 'infer-status'; }, 6000);
    }
}

inferBtn.addEventListener('click', () => runInference(false));

// ── Auto-inference: syncs to the 5-minute candle boundary ──────
function scheduleAutoInference() {
    const now = new Date();
    const ms = (now.getUTCMinutes() % 5) * 60000 + now.getUTCSeconds() * 1000 + now.getUTCMilliseconds();
    const msLeft = 5 * 60000 - ms + 2000; // 2 s after candle close
    console.log(`[AutoInfer] Next inference in ${(msLeft / 1000).toFixed(0)}s`);
    setTimeout(() => {
        runInference(true);
        setInterval(() => runInference(true), 5 * 60 * 1000);
    }, msLeft);
}

// ═══════════════════════════════════════════════════════════════
// Trade History
// ═══════════════════════════════════════════════════════════════
function signalPill(sig) {
    if (!sig || sig === '—') return '<span class="pill">—</span>';
    if (sig === 'BUY UP') return `<span class="pill pill--up">↑ BUY UP</span>`;
    if (sig === 'BUY DOWN') return `<span class="pill pill--down">↓ BUY DOWN</span>`;
    return `<span class="pill pill--hold">— HOLD</span>`;
}

async function loadHistory() {
    try {
        const r = await fetch(`${API}/api/history?n=30`);
        const d = await r.json();
        const trades = (d.trades || []).reverse();

        if (trades.length === 0) {
            historyBody.innerHTML = `<tr><td colspan="8" class="empty-row">No trades yet — run inference to begin.</td></tr>`;
            return;
        }

        historyBody.innerHTML = trades.map(t => {
            const pnl = parseFloat(t.pnl ?? 0);
            const pnlStr = pnl !== 0 ? `<span class="${pnl > 0 ? 'pnl-pos' : 'pnl-neg'}">${pnl > 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}</span>` : '—';
            return `<tr>
        <td>${fmtShortTS(t.timestamp)} UTC</td>
        <td>${fmtPrice(t.btc_price)}</td>
        <td>${signalPill(t.signal)}</td>
        <td>${(parseFloat(t.ensemble_prob || 0) * 100).toFixed(2)}%</td>
        <td>${parseFloat(t.confidence || 0).toFixed(1)}%</td>
        <td>${t.stake > 0 ? '$' + parseFloat(t.stake).toFixed(2) : '—'}</td>
        <td>${pnlStr}</td>
        <td>${fmtPrice(t.virtual_balance)}</td>
      </tr>`;
        }).join('');
    } catch (e) {
        console.error('History load error:', e);
    }
}

$('refreshHistory').addEventListener('click', loadHistory);

// ═══════════════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════════════
async function boot() {
    console.log('[BTC Sentinel] Booting…');

    // Connect WebSocket first (price stream)
    connectWS();

    // Init chart
    initChart();
    await loadCandles(activeLimit);

    // Load history
    await loadHistory();

    // Fetch latest signal (non-blocking, shows stale state if any)
    try {
        const r = await fetch(`${API}/api/balance`);
        const d = await r.json();
        if (d.virtual_balance !== undefined) {
            kpiBalanceVal.textContent = fmtPrice(d.virtual_balance);
        }
    } catch { }

    // Schedule auto-inference at next 5m boundary
    scheduleAutoInference();

    console.log('[BTC Sentinel] Ready.');
}

boot();
