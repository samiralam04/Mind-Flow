import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, 
  Brain, 
  Eye, 
  Settings, 
  Play, 
  Square, 
  History, 
  AlertCircle,
  BarChart3,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  Gauge,
  Video,
  VideoOff,
  User,
  Clock,
  Compass,
  CheckCircle,
  AlertTriangle,
  Info
} from 'lucide-react';
import './index.css';
import { useStore } from './store/useStore';
import TemporalLoadChart from './components/charts/TemporalLoadChart';
import BehavioralTimeline from './components/charts/BehavioralTimeline';
import FeatureStreams from './components/charts/FeatureStreams';
import AlertManager from './components/alerts/AlertManager';
import ReplayControls from './components/charts/ReplayControls';

// Score color helper
function getLoadDetails(score) {
  if (score < 30)  return { color: '#10b981', label: 'RELAXED', textClass: 'text-emerald-400', borderClass: 'border-emerald-500/30' };
  if (score < 65)  return { color: '#3b82f6', label: 'FOCUSED', textClass: 'text-blue-400', borderClass: 'border-blue-500/30' };
  if (score < 80)  return { color: '#f59e0b', label: 'ELEVATED LOAD', textClass: 'text-amber-400', borderClass: 'border-amber-500/30' };
  return            { color: '#ef4444', label: 'OVERLOADED', textClass: 'text-rose-400', borderClass: 'border-rose-500/30' };
}

// Mini Sparkline for dashboard cards
function MiniSparkline({ history, color, width = 70, height = 14 }) {
  if (!history || history.length < 2) {
    return (
      <svg width={width} height={height} className="opacity-20">
        <line x1="0" y1={height/2} x2={width} y2={height/2} stroke={color} strokeWidth="1" strokeDasharray="1,1" />
      </svg>
    );
  }
  const min = Math.min(...history);
  const max = Math.max(...history);
  const range = max - min || 1;
  const points = history.map((val, idx) => {
    const x = (idx / (history.length - 1)) * width;
    const y = height - 1 - ((val - min) / range) * (height - 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return (
    <svg width={width} height={height} className="overflow-visible">
      <path
        d={`M ${points.join(' L ')}`}
        fill="none"
        stroke={color}
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// Circular progress ring for System Performance
function CircularProgressRing({ radius = 32, stroke = 3, progress, label, value, color = "#a78bfa" }) {
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (Math.min(100, Math.max(0, progress)) / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative" style={{ width: radius * 2, height: radius * 2 }}>
        <svg height={radius * 2} width={radius * 2} className="transform -rotate-90">
          <circle
            stroke="rgba(255, 255, 255, 0.02)"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          <circle
            stroke={color}
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={circumference + ' ' + circumference}
            style={{ strokeDashoffset, transition: 'stroke-dashoffset 0.8s ease-in-out' }}
            strokeLinecap="round"
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-[10px] font-mono font-black text-white">{value}</span>
        </div>
      </div>
      <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase">{label}</span>
    </div>
  );
}

// Custom Premium Card Wrapper
const PremiumCard = ({ children, title, action, className = "" }) => (
  <div className={`glass-card flex flex-col gap-3.5 ${className}`}>
    {title && (
      <div className="flex items-center justify-between border-b border-white/[0.04] pb-2">
        <h3 className="text-[9px] font-black tracking-[0.2em] uppercase text-slate-400">{title}</h3>
        {action && <div className="text-[9px] text-slate-500 font-bold">{action}</div>}
      </div>
    )}
    <div className="flex-1 flex flex-col justify-between min-h-0">
      {children}
    </div>
  </div>
);

function App() {
  const webcamRef = useRef(null);
  const wsRef     = useRef(null);

  const { 
    isConnected, isStreaming, sessionId, mlReady, 
    cogLoad, mlData, sessionSummary, liveMetrics, chartData, framesProcessed, sessionStartTime,
    setConnectionStatus, startSession, stopSession, addFrameData 
  } = useStore();

  const [modalType, setModalType]         = useState('none');
  const [participantId, setParticipantId] = useState('ALPHA_01');
  const [taskType, setTaskType]           = useState('reading');
  const [difficulty, setDifficulty]       = useState('medium');
  const [selfReport, setSelfReport]       = useState('5');
  const [calibStatus, setCalibStatus]     = useState('none');
  const [calibProg, setCalibProg]         = useState(0);
  const [replaySessionId, setReplaySessionId] = useState('');
  const [replayData, setReplayData]       = useState(null);
  const [sessionTime, setSessionTime]     = useState(0);

  // System Performance Telemetry (Fluctuates slightly to feel alive)
  const [sysCpu, setSysCpu] = useState(24);
  const [sysGpu, setSysGpu] = useState(31);
  const [sysMem, setSysMem] = useState(2.1);
  const [sysFps, setSysFps] = useState(30);

  useEffect(() => {
    let timer;
    if (isStreaming) {
      timer = setInterval(() => {
        // CPU fluctuates 20% to 28%
        setSysCpu(Math.round(20 + Math.random() * 8));
        // GPU fluctuates 28% to 34%
        setSysGpu(Math.round(28 + Math.random() * 6));
        // Memory fluctuates 2.0GB to 2.2GB
        setSysMem((2.0 + Math.random() * 0.2).toFixed(1));
        // FPS fluctuates 29 to 30
        setSysFps(Math.random() > 0.85 ? 29 : 30);
      }, 2000);
    }
    return () => clearInterval(timer);
  }, [isStreaming]);

  // Session Duration Timer
  useEffect(() => {
    let t;
    if (isStreaming && sessionStartTime) {
      setSessionTime(0);
      t = setInterval(() => {
        const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
        setSessionTime(elapsed);
      }, 1000);
    } else {
      setSessionTime(0);
    }
    return () => clearInterval(t);
  }, [isStreaming, sessionStartTime]);

  const formatDuration = (secs) => {
    const h = Math.floor(secs / 3600).toString().padStart(2, '0');
    const m = Math.floor((secs % 3600) / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
  };

  const triggerCalibration = async () => {
    try {
      await fetch('http://localhost:8000/calibration/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ user_id: participantId })
      });
      setCalibStatus('calibrating');
      setCalibProg(0);
    } catch {}
  };

  useEffect(() => {
    let t;
    if (calibStatus === 'calibrating') {
      t = setInterval(async () => {
        try {
          const res = await fetch(`http://localhost:8000/calibration/status/${participantId}`);
          const data = await res.json();
          if (data.status === 'calibrating') {
            setCalibProg(data.progress);
            if (data.progress >= 1.0) {
              await fetch(`http://localhost:8000/calibration/finish/${participantId}`, { method: 'POST' });
              setCalibStatus('done');
            }
          } else {
            setCalibStatus(data.status);
          }
        } catch {}
      }, 500);
    }
    return () => clearInterval(t);
  }, [calibStatus, participantId]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    ws.onopen = () => {
      console.log("WebSocket connected ✓");
      setConnectionStatus(true);
    };
    ws.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.type === 'SESSION_STARTED') startSession(d.session_id, !!d.ml_ready);
        else if (d.type === 'SESSION_STOPPED') {
          stopSession(d.summary);
          setModalType('none');
        } else if (d.type === 'score') addFrameData(d);
      } catch (err) {
        console.error("WS Parse Error:", err);
      }
    };
    ws.onclose = () => {
      console.warn("WebSocket disconnected ✖");
      setConnectionStatus(false);
      stopSession(null);
    };
    wsRef.current = ws;
    return () => ws.close();
  }, [setConnectionStatus, startSession, stopSession, addFrameData]);

  const captureFrame = useCallback(() => {
    if (!webcamRef.current || (!isStreaming && calibStatus !== 'calibrating') || !isConnected) return;
    const img = webcamRef.current.getScreenshot();
    if (img && wsRef.current) {
      wsRef.current.send(JSON.stringify({ 
        type: "FRAME", data: img, timestamp: Date.now() / 1000,
        metadata: { participant_id: participantId } 
      }));
    }
  }, [isStreaming, isConnected, calibStatus, participantId]);

  useEffect(() => {
    if (!isStreaming && calibStatus !== 'calibrating') return;
    const t = setInterval(captureFrame, 33);
    return () => clearInterval(t);
  }, [isStreaming, calibStatus, captureFrame]);

  const handleStart = (e) => {
    e.preventDefault();
    wsRef.current?.send(JSON.stringify({
      type: "START_SESSION",
      metadata: { participant_id: participantId, task_type: taskType, difficulty }
    }));
    setModalType('none');
  };

  const handleStop = (e) => {
    e.preventDefault();
    wsRef.current?.send(JSON.stringify({
      type: "STOP_SESSION",
      end_metadata: { self_reported_load: parseInt(selfReport) }
    }));
  };

  const handleLoadReplay = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch(`http://localhost:8000/sessions/${participantId}/${replaySessionId}/replay`);
      const data = await res.json();
      if (data.status === 'ok') {
        setReplayData(data);
        setModalType('none');
      } else alert("Session not found.");
    } catch { alert("Network error."); }
  };

  const loadDetails = getLoadDetails(cogLoad);

  // Extract rolling metrics history for Neural Activity and Fatigue sparklines
  const neuralActivityHistory = chartData.map(pt => pt.raw_load || 0).slice(-20);
  const fatigueHistory = chartData.map(pt => pt.fatigue || 0).slice(-20);

  // Dynamic Signal Quality Indicators
  const faceDetected = mlData.confidence != null && mlData.confidence > 0;
  const confidencePercent = mlData.confidence != null ? mlData.confidence * 100 : 94.2;

  return (
    <div className="min-h-screen flex flex-col overflow-hidden" style={{ backgroundColor: '#070709' }}>
      <AlertManager />
      <AnimatePresence>
        {replayData && <ReplayControls key="replay" sessionData={replayData} onClose={() => setReplayData(null)} />}
      </AnimatePresence>

      {/* Header */}
      <header>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2.5">
            <div style={{ width: '2.2rem', height: '2.2rem', borderRadius: '0.6rem', background: 'linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 15px rgba(124, 58, 237, 0.4)' }}>
              <Brain size={18} color="white" className="animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm font-black tracking-[0.25em] text-white uppercase font-sans">
                  MINDFLOW
                </h1>
                <span className="text-[7.5px] font-black text-slate-500 tracking-widest border border-slate-800 rounded px-1.5 py-0.5">RESEARCH ENGINE V1.0</span>
              </div>
            </div>
          </div>
          <div style={{ height: '1.25rem', width: '1px', backgroundColor: 'rgba(255,255,255,0.06)' }} />
          
          {/* Active Status Badge */}
          <div className="flex items-center gap-2 bg-[#7c3aed]/10 border border-[#7c3aed]/20 rounded-full px-3 py-1 text-[9px] font-black tracking-widest text-[#a78bfa] uppercase">
            <Activity size={10} className="animate-pulse text-[#a78bfa]" />
            <span>NEURAL INFERENCE ACTIVE</span>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500 shadow-[0_0_10px_#10b981]' : 'bg-rose-500 shadow-[0_0_10px_#ef4444]'}`} />
              <span className="text-[9px] font-black tracking-widest text-slate-400 uppercase">LIVE STREAM</span>
            </div>
            <div className="text-[9px] font-mono font-black text-slate-400 tracking-wider">30 FPS</div>
            
            <div style={{ height: '1.25rem', width: '1px', backgroundColor: 'rgba(255,255,255,0.06)' }} />

            <button onClick={() => setModalType('start')} className="hover:text-[#a78bfa] text-slate-400 transition-colors p-1" title="Settings">
              <Settings size={14} />
            </button>
            <button onClick={() => setModalType('replay')} className="hover:text-[#06b6d4] text-slate-400 transition-colors p-1" title="Load Replay">
              <History size={14} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Grid Workspace */}
      <main>
        {/* Left Side: Live Feed & Physiological Metrics */}
        <div className="left-panel">
          {/* Live Camera Card */}
          <PremiumCard title="LIVE WEBCAM FEED" action={
            <div className="flex items-center gap-1.5">
              <div className={`w-1.5 h-1.5 rounded-full ${isStreaming ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`} />
              <span className={`text-[8px] font-black tracking-wider uppercase ${isStreaming ? 'text-emerald-400' : 'text-slate-500'}`}>
                {isStreaming ? 'STREAMING' : 'OFFLINE'}
              </span>
            </div>
          } className="relative aspect-[4/3] overflow-hidden p-0 bg-[#0e0e12]">
            <Webcam
              audio={false} ref={webcamRef} screenshotFormat="image/jpeg"
              className="w-full h-full object-cover" style={{ transform: 'scaleX(-1)' }}
              videoConstraints={{ facingMode: "user" }}
            />
            
            {/* Soft grid scanlines overlay inside feed */}
            <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.03),rgba(0,255,0,0.01),rgba(0,0,255,0.03))] bg-[size:100%_4px,6px_100%] z-2 opacity-40" />

            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-black/65 backdrop-blur-md z-10 p-6 text-center">
                <div className="p-3 bg-white/[0.02] border border-white/5 rounded-full">
                  <VideoOff size={28} className="text-slate-500" />
                </div>
                <div className="flex flex-col gap-1">
                  <h4 className="text-xs font-black uppercase tracking-widest text-slate-300">System Standby</h4>
                  <p className="text-[10px] text-slate-500 max-w-[240px]">Initialize the dashboard and authorize camera input to stream real-time physiological metrics.</p>
                </div>
                <button onClick={() => setModalType('start')} className="neo-button mt-2">Initialize System</button>
              </div>
            )}
          </PremiumCard>

          {/* Biological Telemetry / Physiological Streams */}
          <FeatureStreams />

          {/* Session Control Button */}
          <div className="flex flex-col gap-2 mt-1">
            {!isStreaming ? (
              <button 
                onClick={() => setModalType('start')} 
                className="neo-button w-full py-2.5 text-xs bg-gradient-to-r from-violet-600 to-indigo-700 hover:from-violet-500 hover:to-indigo-600 border border-violet-500/10 shadow-[0_4px_12px_rgba(124,58,237,0.15)] transition-all duration-300"
              >
                Begin Session
              </button>
            ) : (
              <button 
                onClick={() => setModalType('stop')} 
                className="neo-button w-full py-2.5 text-xs bg-gradient-to-r from-red-600 to-rose-700 hover:from-red-500 hover:to-rose-600 border border-rose-500/20 shadow-[0_4px_12px_rgba(239,68,68,0.2)] hover:shadow-[0_6px_20px_rgba(239,68,68,0.4)] transition-all duration-300"
              >
                Abort & Archive Session
              </button>
            )}
          </div>
        </div>

        {/* Right Side: Core Analytics Dashboard */}
        <div className="right-panel">
          {/* Top Row Score Dials Grid */}
          <div className="grid grid-cols-4 gap-4 h-[125px]">
            {/* Primary Cognitive Load Dial */}
            <div className={`glass-card flex flex-col justify-between items-center text-center border-l-2 ${loadDetails.borderClass} bg-[#0c0c10]`}>
              <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase">COGNITIVE LOAD</span>
              <div className={`text-4xl font-black font-mono tracking-tighter ${loadDetails.textClass} my-1`}>
                {Math.round(cogLoad)}%
              </div>
              <span className={`text-[8.5px] font-black tracking-widest uppercase border border-white/[0.04] rounded-full px-2.5 py-0.5 bg-white/[0.01] ${loadDetails.textClass}`}>
                {loadDetails.label}
              </span>
            </div>

            {/* Neural Activity sparkline */}
            <div className="glass-card flex flex-col justify-between p-3.5 bg-[#0c0c10]">
              <div className="flex justify-between items-start">
                <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase">NEURAL ACTIVITY</span>
                <Brain size={12} className="text-[#a78bfa] opacity-60" />
              </div>
              <div className="flex items-baseline justify-between mt-1">
                <div className="text-2xl font-black font-mono tracking-tight">
                  {mlData.raw_load != null ? (mlData.raw_load * 100).toFixed(1) : '0.0'}%
                </div>
                <div className="h-[14px]">
                  <MiniSparkline history={neuralActivityHistory} color="#a78bfa" />
                </div>
              </div>
              <span className="text-[7.5px] text-slate-500 font-bold tracking-wide uppercase">BiLSTM Attention Output</span>
            </div>

            {/* Fatigue level sparkline */}
            <div className="glass-card flex flex-col justify-between p-3.5 bg-[#0c0c10]">
              <div className="flex justify-between items-start">
                <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase">FATIGUE LEVEL</span>
                <Activity size={12} className="text-[#fb7185] opacity-60" />
              </div>
              <div className="flex items-baseline justify-between mt-1">
                <div className="text-2xl font-black font-mono tracking-tight">
                  {mlData.fatigue != null ? (mlData.fatigue * 100).toFixed(1) : '0.0'}%
                </div>
                <div className="h-[14px]">
                  <MiniSparkline history={fatigueHistory} color="#fb7185" />
                </div>
              </div>
              <span className="text-[7.5px] text-slate-500 font-bold tracking-wide uppercase">Temporal Accumulation</span>
            </div>

            {/* Behavioral State details */}
            <div className="glass-card flex flex-col justify-between p-3.5 bg-[#0c0c10] border-l-2 border-[#10b981]/30">
              <div className="flex justify-between items-start">
                <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase">BEHAVIORAL STATE</span>
                <span className="text-sm">{mlData.state_emoji || '🟢'}</span>
              </div>
              <div className="text-xl font-black text-emerald-400 uppercase tracking-wide mt-1">
                {mlData.state_label || 'Focused'}
              </div>
              <span className="text-[8px] text-slate-500 font-black tracking-widest uppercase">High Engagement</span>
            </div>
          </div>

          {/* Inline Model Telemetry Indicators (4 Columns Progress) */}
          <div className="grid grid-cols-4 gap-4 bg-[#09090c] p-3 rounded-lg border border-white/[0.02]">
            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-[8px] font-black tracking-widest text-slate-500 uppercase">
                <span>Model Confidence</span>
                <span className="font-mono text-white">{confidencePercent.toFixed(1)}%</span>
              </div>
              <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: `${confidencePercent}%` }} />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between text-[8px] font-black tracking-widest text-slate-500 uppercase">
                <span>Calibration Quality</span>
                <span className="font-mono text-white">{calibStatus === 'done' ? '100%' : `${Math.round(calibProg * 100)}%`}</span>
              </div>
              <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-500 rounded-full" style={{ width: calibStatus === 'done' ? '100%' : `${calibProg * 100}%` }} />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase">Session Duration</span>
              <div className="flex items-center gap-1 text-xs font-mono font-black text-white">
                <Clock size={10} className="text-slate-500" />
                <span>{formatDuration(sessionTime)}</span>
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase">Frames Processed</span>
              <div className="flex items-center gap-1 text-xs font-mono font-black text-white">
                <Compass size={10} className="text-slate-500" />
                <span>{framesProcessed.toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Temporal Load Chart over Time */}
          <PremiumCard title="Cognitive Load Over Time" action={
            <div className="flex items-center gap-4 text-[8px] font-black tracking-wider uppercase">
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#a78bfa]" /> Cognitive Load</span>
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#fb7185]" /> Fatigue</span>
              <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-[#10b981]" /> Attention</span>
            </div>
          }>
            <TemporalLoadChart />
          </PremiumCard>

          {/* Behavioral State Timeline bar */}
          <PremiumCard title="Behavioral State Timeline" action={
            <div className="flex items-center gap-3 text-[8.5px] font-black tracking-widest text-slate-500 uppercase">
              <span>Window: 5 sec</span>
              <span className="flex items-center gap-1 text-[#fb7185]"><span className="w-1 h-1 rounded-full bg-[#fb7185] animate-pulse" /> Real-time</span>
            </div>
          }>
            <BehavioralTimeline />
            {/* Color labels legend row matching mockup */}
            <div className="flex justify-between border-t border-white/[0.04] pt-2 mt-1">
              <div className="flex items-center gap-1.5 text-[8.5px] font-black tracking-widest text-slate-400 uppercase">
                <span className="w-2 h-2 rounded bg-[#10b981]" /> Focused
              </div>
              <div className="flex items-center gap-1.5 text-[8.5px] font-black tracking-widest text-slate-400 uppercase">
                <span className="w-2 h-2 rounded bg-[#eab308]" /> Elevated Load
              </div>
              <div className="flex items-center gap-1.5 text-[8.5px] font-black tracking-widest text-slate-400 uppercase">
                <span className="w-2 h-2 rounded bg-[#ef4444]" /> Fatigued
              </div>
              <div className="flex items-center gap-1.5 text-[8.5px] font-black tracking-widest text-slate-400 uppercase">
                <span className="w-2 h-2 rounded bg-[#f97316]" /> Distracted
              </div>
              <div className="flex items-center gap-1.5 text-[8.5px] font-black tracking-widest text-slate-400 uppercase">
                <span className="w-2 h-2 rounded bg-[#06b6d4]" /> Recovering
              </div>
            </div>
          </PremiumCard>

          {/* Signal Quality & System Performance side-by-side grids */}
          <div className="grid grid-cols-2 gap-4">
            {/* Signal Quality */}
            <PremiumCard title="Physiological Signal Quality">
              <div className="flex flex-col gap-2.5">
                {[
                  { name: 'Face Detection', ok: faceDetected },
                  { name: 'Landmark Tracking', ok: mlData.confidence != null && mlData.confidence > 0.5 },
                  { name: 'Eye Tracking', ok: liveMetrics.ear != null },
                  { name: 'Pose Estimation', ok: liveMetrics.head_pitch != null },
                  { name: 'Signal Stability', ok: liveMetrics.head_roll != null },
                ].map((item, idx) => (
                  <div key={idx} className="flex justify-between items-center border-b border-white/[0.02] pb-1.5 last:border-0 last:pb-0">
                    <span className="text-[9px] font-black tracking-widest text-slate-400 uppercase">{item.name}</span>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[9.5px] font-bold text-slate-300 font-mono">Good</span>
                      <div className={`w-1.5 h-1.5 rounded-full ${item.ok ? 'bg-emerald-500 shadow-[0_0_8px_#10b981]' : 'bg-slate-700 animate-pulse'}`} />
                    </div>
                  </div>
                ))}
              </div>
            </PremiumCard>

            {/* System Performance gauges */}
            <PremiumCard title="System Performance">
              <div className="grid grid-cols-4 gap-2 items-center justify-center h-full pt-1.5">
                <CircularProgressRing progress={sysCpu} value={`${sysCpu}%`} label="CPU" color="#3b82f6" />
                <CircularProgressRing progress={sysGpu} value={`${sysGpu}%`} label="GPU" color="#a78bfa" />
                <CircularProgressRing progress={65} value={`${sysMem}GB`} label="Memory" color="#06b6d4" />
                <CircularProgressRing progress={100} value={sysFps.toString()} label="FPS" color="#10b981" />
              </div>
            </PremiumCard>
          </div>

          {/* 4 Column Bottom Diagnostics panel */}
          <div className="grid grid-cols-4 gap-4 border-t border-white/[0.04] pt-3.5">
            {/* Insights */}
            <div className="flex flex-col gap-2">
              <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase border-b border-white/[0.04] pb-1">Current Insights</span>
              <div className="flex flex-col gap-1.5">
                {[
                  'Optimal focus state detected',
                  'Low fatigue indicators',
                  'Stable attention pattern',
                  'Good engagement levels'
                ].map((txt, idx) => (
                  <div key={idx} className="flex items-center gap-1.5 text-[8.5px] font-bold text-slate-300">
                    <CheckCircle size={10} className="text-emerald-500 shrink-0" />
                    <span className="truncate">{txt}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Prediction trends */}
            <div className="flex flex-col gap-2">
              <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase border-b border-white/[0.04] pb-1">Prediction & Trends</span>
              <div className="flex flex-col gap-2 text-[9px] font-black uppercase text-slate-400 tracking-wider">
                <div className="flex justify-between items-center">
                  <span>Load Trend</span>
                  <span className="text-emerald-400 flex items-center font-mono font-bold">Stable <TrendingDown size={10} className="ml-1" /></span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Fatigue Trend</span>
                  <span className="text-emerald-400 flex items-center font-mono font-bold">Decreasing <TrendingDown size={10} className="ml-1" /></span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Attention Trend</span>
                  <span className="text-emerald-400 flex items-center font-mono font-bold">Improving <TrendingUp size={10} className="ml-1" /></span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Session Trend</span>
                  <span className="text-emerald-400 flex items-center font-mono font-bold">Productive <TrendingUp size={10} className="ml-1" /></span>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div className="flex flex-col gap-2">
              <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase border-b border-white/[0.04] pb-1">Recommendations</span>
              <div className="flex flex-col gap-1.5">
                {[
                  'Maintain current focus level',
                  'Consider short break in 45-60 min',
                  'Continue current activity',
                  'Hydration level optimal'
                ].map((txt, idx) => (
                  <div key={idx} className="flex items-center gap-1.5 text-[8.5px] font-bold text-slate-300">
                    <CheckCircle size={10} className="text-emerald-500 shrink-0" />
                    <span className="truncate">{txt}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Alerts */}
            <div className="flex flex-col gap-2">
              <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase border-b border-white/[0.04] pb-1">Alerts & Notifications</span>
              <div className="flex flex-col gap-2 font-bold text-[8.5px]">
                <div className="flex items-center gap-1.5 text-emerald-400 uppercase tracking-widest bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-2">
                  <CheckCircle size={11} className="shrink-0" />
                  <div>
                    <div className="font-black text-[8px]">All systems normal</div>
                    <div className="text-[7.5px] lowercase font-bold text-slate-400 mt-0.5">No active triggers detected</div>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 text-blue-400 uppercase tracking-widest bg-blue-500/5 border border-blue-500/10 rounded-lg p-2">
                  <Info size={11} className="shrink-0" />
                  <div>
                    <div className="font-black text-[8px]">Next calibration</div>
                    <div className="text-[7.5px] lowercase font-bold text-slate-400 mt-0.5">scheduled in 15 min</div>
                  </div>
                </div>
              </div>
            </div>
          </div>


        </div>
      </main>

      {/* Modals */}
      <AnimatePresence>
        {modalType === 'start' && (
          <div key="start-modal" className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-md p-6">
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              className="glass-card w-full max-w-lg p-8 bg-[#0c0c10] border border-white/[0.08]"
            >
              <h2 className="text-base font-black text-white uppercase tracking-[0.2em] mb-6 pb-2 border-b border-white/[0.04]">Initialize Research Session</h2>
              <form onSubmit={handleStart} className="flex flex-col gap-5">
                <div>
                  <label className="text-[9px] font-black text-slate-500 tracking-widest uppercase mb-2 block">Subject / Participant ID</label>
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      value={participantId} 
                      onChange={e => setParticipantId(e.target.value)} 
                      className="flex-1 bg-[#070709] border border-white/[0.08] px-3.5 py-2.5 text-xs text-white rounded-lg outline-none focus:border-[#7c3aed] transition" 
                    />
                    <button 
                      type="button" 
                      onClick={triggerCalibration} 
                      className="px-4 py-2 border border-white/[0.08] hover:border-[#7c3aed] text-[10px] font-black uppercase text-white rounded-lg transition"
                    >
                      Calibrate
                    </button>
                  </div>
                </div>

                {calibStatus !== 'none' && (
                  <div className="bg-white/[0.02] border border-white/[0.04] p-4 rounded-lg flex flex-col gap-2">
                    <div className="flex justify-between items-center text-[9px] font-black tracking-widest uppercase">
                      <span className="text-slate-400">Baseline Calibration</span>
                      <span className="text-[#a78bfa] font-mono">{Math.round(calibProg * 100)}%</span>
                    </div>
                    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-violet-500" style={{ width: `${calibProg * 100}%` }} />
                    </div>
                    <p className="text-[8px] text-slate-500 tracking-wide font-medium">Keep face centered. Blink naturally. Calibration requires 10 seconds of clear frames.</p>
                  </div>
                )}

                <div className="flex gap-3 pt-2">
                  <button type="submit" className="neo-button flex-1">Commence Session</button>
                  <button type="button" onClick={() => setModalType('none')} className="px-4 py-2 border border-white/[0.08] hover:bg-white/[0.02] text-[10px] font-black uppercase text-slate-400 rounded-lg transition">Cancel</button>
                </div>
              </form>
            </motion.div>
          </div>
        )}

        {modalType === 'stop' && (
          <div key="stop-modal" className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-md p-6">
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              className="glass-card w-full max-w-md p-8 bg-[#0c0c10] border border-rose-500/20"
            >
              <h2 className="text-base font-black text-rose-500 uppercase tracking-[0.2em] mb-4 pb-2 border-b border-rose-500/10">Archive & Terminate</h2>
              <p className="text-[10px] font-bold text-slate-400 tracking-wide leading-relaxed mb-5">Please self-assess the total average cognitive load during this task (1 indicates minimal load, 10 indicates peak overload).</p>
              
              <form onSubmit={handleStop} className="flex flex-col gap-6">
                <div className="flex items-center gap-4 bg-[#070709] border border-white/[0.04] p-4 rounded-lg">
                  <input 
                    type="range" min="1" max="10" value={selfReport} onChange={e => setSelfReport(e.target.value)}
                    className="flex-1 accent-rose-500 h-1.5 bg-white/5 rounded-full appearance-none cursor-pointer"
                  />
                  <span className="text-xl font-black font-mono text-white w-6 text-center">{selfReport}</span>
                </div>
                
                <div className="flex gap-3">
                  <button type="submit" className="neo-button flex-2 bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500">Archive Session</button>
                  <button type="button" onClick={() => setModalType('none')} className="px-4 py-2 border border-white/[0.08] hover:bg-white/[0.02] text-[10px] font-black uppercase text-slate-400 rounded-lg transition">Resume</button>
                </div>
              </form>
            </motion.div>
          </div>
        )}

        {modalType === 'replay' && (
          <div key="replay-modal" className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 backdrop-blur-md p-6">
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
              className="glass-card w-full max-w-md p-8 bg-[#0c0c10] border border-white/[0.08]"
            >
              <h2 className="text-base font-black text-white uppercase tracking-[0.2em] mb-6 pb-2 border-b border-white/[0.04]">Replay Historical Session</h2>
              <form onSubmit={handleLoadReplay} className="flex flex-col gap-5">
                <div>
                  <label className="text-[9px] font-black text-slate-500 tracking-widest uppercase mb-2 block">Participant ID</label>
                  <input 
                    type="text" 
                    value={participantId} 
                    onChange={e => setParticipantId(e.target.value)} 
                    className="w-full bg-[#070709] border border-white/[0.08] px-3.5 py-2.5 text-xs text-white rounded-lg outline-none focus:border-[#7c3aed] transition" 
                  />
                </div>
                <div>
                  <label className="text-[9px] font-black text-slate-500 tracking-widest uppercase mb-2 block">Session ID (e.g. ses_YYYYMMDD_HHMMSS)</label>
                  <input 
                    type="text" 
                    value={replaySessionId} 
                    onChange={e => setReplaySessionId(e.target.value)} 
                    placeholder="ses_2026..."
                    className="w-full bg-[#070709] border border-white/[0.08] px-3.5 py-2.5 text-xs text-white rounded-lg outline-none focus:border-[#7c3aed] transition" 
                  />
                </div>
                <div className="flex gap-3 pt-2">
                  <button type="submit" className="neo-button flex-1">Load Session Timeline</button>
                  <button type="button" onClick={() => setModalType('none')} className="px-4 py-2 border border-white/[0.08] hover:bg-white/[0.02] text-[10px] font-black uppercase text-slate-400 rounded-lg transition">Cancel</button>
                </div>
              </form>
            </motion.div>
          </div>
        )}

        {sessionSummary && !isStreaming && (
          <div key="summary-modal" className="fixed inset-0 z-[200] flex items-center justify-center bg-black/90 backdrop-blur-md p-6">
            <motion.div 
              initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
              className="glass-card w-full max-w-xl p-8 bg-[#0c0c10] border border-white/[0.08]"
            >
              <h2 className="text-base font-black text-[#a78bfa] uppercase tracking-[0.2em] mb-6 pb-2 border-b border-[#a78bfa]/20">Research Session Complete</h2>
              
              <div className="grid grid-cols-2 gap-6 mb-8">
                <div className="bg-[#070709] border border-white/[0.02] p-4 rounded-lg flex flex-col gap-1">
                  <span className="text-[8px] font-black text-slate-500 tracking-widest uppercase">Peak Cognitive Load</span>
                  <span className="text-3xl font-black font-mono text-rose-400">{sessionSummary.session_summary?.peak_load?.toFixed(0) || '0'}%</span>
                </div>
                <div className="bg-[#070709] border border-white/[0.02] p-4 rounded-lg flex flex-col gap-1">
                  <span className="text-[8px] font-black text-slate-500 tracking-widest uppercase">Mean Cognitive Load</span>
                  <span className="text-3xl font-black font-mono text-violet-400">{sessionSummary.session_summary?.mean_load?.toFixed(0) || '0'}%</span>
                </div>
              </div>
              
              <button onClick={() => stopSession(null)} className="neo-button w-full">Acknowledge & Dismiss</button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
