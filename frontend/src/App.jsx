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
  Gauge
} from 'lucide-react';
import './index.css';
import { useStore } from './store/useStore';
import TemporalLoadChart from './components/charts/TemporalLoadChart';
import BehavioralTimeline from './components/charts/BehavioralTimeline';
import FeatureStreams from './components/charts/FeatureStreams';
import AlertManager from './components/alerts/AlertManager';
import ReplayControls from './components/charts/ReplayControls';

// Score colour helpers
function getScoreColor(score) {
  if (score < 25)  return { text: '#10b981', label: 'Flow State', emoji: '🟢' };
  if (score < 50)  return { text: '#3b82f6', label: 'Engaged',   emoji: '🔵' };
  if (score < 70)  return { text: '#eab308', label: 'High Focus', emoji: '🟡' };
  if (score < 85)  return { text: '#f97316', label: 'Near Peak',  emoji: '🟠' };
  return            { text: '#ef4444', label: 'Overloaded', emoji: '🔴' };
}

// Solid Card Component
const Card = ({ children, title, icon: Icon, className = "" }) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className={`glass-card flex flex-col gap-4 ${className}`}
    style={{ position: 'relative', zIndex: 1 }}
  >
    {title && (
      <div className="flex items-center justify-between" style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.75rem' }}>
        <div className="flex items-center gap-2">
          {Icon && <Icon size={16} style={{ color: '#a78bfa' }} />}
          <h3 style={{ fontSize: '11px', fontWeight: '900', letterSpacing: '0.2em', textTransform: 'uppercase', color: '#cbd5e1', margin: 0 }}>{title}</h3>
        </div>
      </div>
    )}
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      {children}
    </div>
  </motion.div>
);

// ML status badge
function MLBadge({ ml }) {
  const statusStyle = {
    padding: '0.25rem 0.75rem',
    borderRadius: '9999px',
    fontSize: '10px',
    fontWeight: '900',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    border: '1px solid rgba(255,255,255,0.1)',
    backgroundColor: 'rgba(255,255,255,0.05)'
  };

  if (!ml || ml.status === 'heuristic') {
    return (
      <div style={statusStyle}>
        <Activity size={12} />
        <span>HEURISTIC ENGINE</span>
      </div>
    );
  }
  
  return (
    <div style={{ ...statusStyle, color: '#a78bfa', borderColor: 'rgba(167, 139, 250, 0.3)' }}>
      <Brain size={12} className={ml.status === 'predicting' ? 'animate-pulse' : ''} />
      <span>
        {ml.status === 'predicting' ? `NEURAL INFERENCE (${ml.inference_ms?.toFixed(0)}ms)` :
         ml.status === 'warming_up' ? `SEQUENCING BUFFER` :
         `WAITING FOR SUBJECT`}
      </span>
    </div>
  );
}

function App() {
  const webcamRef = useRef(null);
  const wsRef     = useRef(null);

  const { 
    isConnected, isStreaming, sessionId, mlReady, 
    cogLoad, mlData, sessionSummary,
    setConnectionStatus, startSession, stopSession, addFrameData 
  } = useStore();

  const [modalType,  setModalType]      = useState('none');
  const [participantId, setParticipantId] = useState('ALPHA_01');
  const [taskType,      setTaskType]      = useState('reading');
  const [difficulty,    setDifficulty]    = useState('medium');
  const [selfReport,    setSelfReport]    = useState('5');
  const [calibStatus, setCalibStatus] = useState('none');
  const [calibProg,   setCalibProg]   = useState(0);
  const [replaySessionId, setReplaySessionId] = useState('');
  const [replayData, setReplayData]           = useState(null);

  const checkCalibration = useCallback(async (uid) => {
    try {
      const res = await fetch(`http://localhost:8000/calibration/status/${uid}`);
      const data = await res.json();
      setCalibStatus(data.status);
      if (data.status === 'calibrating') setCalibProg(data.progress);
    } catch {}
  }, []);

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
    console.log("Starting session for", participantId);
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

  const sc = getScoreColor(cogLoad);

  return (
    <div className="min-h-screen flex flex-col overflow-hidden" style={{ backgroundColor: '#0a0a0c' }}>
      <AlertManager />
      <AnimatePresence>
        {replayData && <ReplayControls key="replay" sessionData={replayData} onClose={() => setReplayData(null)} />}
      </AnimatePresence>

      {/* Dashboard Header */}
      <header>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div style={{ width: '2rem', height: '2rem', borderRadius: '0.5rem', background: 'linear-gradient(to bottom right, #7c3aed, #4f46e5)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Brain size={18} color="white" />
            </div>
            <div>
              <h1 style={{ fontSize: '14px', fontWeight: '900', letterSpacing: '0.2em', textTransform: 'uppercase', margin: 0, color: 'white' }}>
                MindFlow
              </h1>
              <p style={{ fontSize: '9px', fontWeight: '900', color: '#a78bfa', letterSpacing: '0.1em', textTransform: 'uppercase', margin: 0 }}>Research Env v6.2</p>
            </div>
          </div>
          <div style={{ height: '1rem', width: '1px', backgroundColor: 'rgba(255,255,255,0.1)' }} />
          <MLBadge ml={mlData} />
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-4">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '0.6rem', height: '0.6rem', borderRadius: '50%', backgroundColor: isConnected ? '#10b981' : '#ef4444', boxShadow: isConnected ? '0 0 10px #10b981' : 'none' }} />
              <span style={{ fontSize: '11px', fontWeight: '900', color: '#94a3b8' }}>{isConnected ? 'ONLINE' : 'OFFLINE'}</span>
            </div>
            <button onClick={() => setModalType('start')} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.5rem', color: '#94a3b8' }}>
              <Settings size={18} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main>
        {/* Left Col: Visual Telemetry */}
        <div className="webcam-container flex flex-col gap-6">
          <Card className="relative" style={{ aspectRatio: '16/9', padding: 0, overflow: 'hidden' }}>
            <Webcam
              audio={false} ref={webcamRef} screenshotFormat="image/jpeg"
              className="w-full h-full" style={{ objectFit: 'cover', transform: 'scaleX(-1)' }}
              videoConstraints={{ facingMode: "user" }}
            />
            {!isStreaming && (
              <div className="absolute inset-0 flex items-center justify-center" style={{ backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)', zIndex: 10 }}>
                <button onClick={() => { console.log("Init clicked"); setModalType('start'); }} className="neo-button">Initialize System</button>
              </div>
            )}
          </Card>
          <FeatureStreams />
        </div>

        {/* Right Col: Analytics */}
        <div className="flex-1 flex flex-col gap-6">
          <div className="flex gap-6" style={{ height: '220px' }}>
            {/* Primary Score Dial */}
            <Card className="flex items-center justify-center text-center" style={{ flex: 2, backgroundColor: 'rgba(255,255,255,0.02)' }}>
              <div style={{ fontSize: '80px', fontWeight: '900', color: sc.text, letterSpacing: '-0.05em' }}>
                {Math.round(cogLoad)}
              </div>
              <div style={{ marginTop: '0.5rem', padding: '0.25rem 1rem', borderRadius: '9999px', border: `1px solid ${sc.text}`, color: sc.text, fontSize: '10px', fontWeight: '900', textTransform: 'uppercase' }}>
                {sc.label}
              </div>
            </Card>

            {/* Neural Summary Grid */}
            <div style={{ flex: 3, display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr', gap: '1rem' }}>
              {[
                { label: 'Neural Activity', val: mlData.raw_load != null ? (mlData.raw_load * 100).toFixed(1) + '%' : '—', icon: Brain, color: '#a78bfa' },
                { label: 'Fatigue Drain', val: mlData.fatigue != null ? (mlData.fatigue * 100).toFixed(1) + '%' : '—', icon: Activity, color: '#fb7185' },
                { label: 'Capture Conf.', val: mlData.confidence != null ? (mlData.confidence * 100).toFixed(0) + '%' : '—', icon: Eye, color: '#3b82f6' },
                { label: 'Behavioral State', val: mlData.state_label || 'Standby', icon: History, color: '#10b981' },
              ].map((item) => (
                <div key={item.label} className="glass-card" style={{ padding: '1rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ color: item.color }}><item.icon size={18} /></div>
                  <div>
                    <p style={{ fontSize: '8px', fontWeight: '900', color: '#64748b', textTransform: 'uppercase', margin: 0 }}>{item.label}</p>
                    <p style={{ fontSize: '14px', fontWeight: '900', margin: 0 }}>{item.val}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <Card title="Temporal Performance Log" icon={Activity} style={{ flex: 1 }}>
            <TemporalLoadChart />
          </Card>

          <Card title="Cognitive State Transitions" icon={History} style={{ height: '100px' }}>
            <BehavioralTimeline />
          </Card>
        </div>
      </main>

      {/* Control Bar */}
      <footer>
        {!isStreaming ? (
          <button onClick={() => setModalType('start')} className="neo-button" style={{ padding: '0.75rem 4rem' }}>Begin Mission</button>
        ) : (
          <button onClick={() => setModalType('stop')} className="neo-button" style={{ backgroundColor: '#ef4444', padding: '0.75rem 4rem' }}>Abort & Archive</button>
        )}
      </footer>

      {/* Modals */}
      <AnimatePresence>
        {modalType === 'start' && (
          <div key="start-modal" className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 p-6">
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
              className="glass-card w-full max-w-lg p-10"
              style={{ backgroundColor: '#121216', borderColor: 'rgba(255,255,255,0.1)' }}
            >
              <h2 style={{ fontSize: '24px', fontWeight: '900', color: 'white', textTransform: 'uppercase', marginBottom: '2rem' }}>Initialize System</h2>
              <form onSubmit={handleStart} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div>
                  <label style={{ fontSize: '10px', fontWeight: '900', color: '#64748b', textTransform: 'uppercase', marginBottom: '0.5rem', display: 'block' }}>Subject ID</label>
                  <div className="flex gap-2">
                    <input type="text" value={participantId} onChange={e => setParticipantId(e.target.value)} 
                      style={{ flex: 1, background: '#0a0a0c', border: '1px solid rgba(255,255,255,0.1)', padding: '0.75rem', color: 'white', borderRadius: '0.5rem' }} />
                    <button type="button" onClick={triggerCalibration} className="neo-button" style={{ fontSize: '10px' }}>Calibrate</button>
                  </div>
                </div>
                <div className="flex gap-4 pt-4">
                  <button type="submit" className="neo-button" style={{ flex: 1 }}>Commence Sync</button>
                  <button type="button" onClick={() => setModalType('none')} style={{ flex: 1, background: 'none', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '0.5rem' }}>Cancel</button>
                </div>
              </form>
            </motion.div>
          </div>
        )}

        {modalType === 'stop' && (
          <div key="stop-modal" className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 p-6">
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
              className="glass-card w-full max-w-md p-10"
              style={{ backgroundColor: '#121216', borderColor: 'rgba(239, 68, 68, 0.3)' }}
            >
              <h2 style={{ fontSize: '20px', fontWeight: '900', color: '#ef4444', textTransform: 'uppercase', marginBottom: '1.5rem' }}>Terminate Mission</h2>
              <p style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '1.5rem' }}>Please provide a self-assessment of your cognitive load during this task (1-10).</p>
              
              <form onSubmit={handleStop} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <input 
                    type="range" min="1" max="10" value={selfReport} onChange={e => setSelfReport(e.target.value)}
                    style={{ flex: 1, accentColor: '#ef4444' }}
                  />
                  <span style={{ fontSize: '18px', fontWeight: '900', color: 'white', width: '2rem' }}>{selfReport}</span>
                </div>
                
                <div className="flex gap-4 pt-4">
                  <button type="submit" className="neo-button" style={{ flex: 2, backgroundColor: '#ef4444' }}>Abort & Archive</button>
                  <button type="button" onClick={() => setModalType('none')} style={{ flex: 1, background: 'none', border: '1px solid rgba(255,255,255,0.1)', color: 'white', borderRadius: '0.5rem' }}>Resume</button>
                </div>
              </form>
            </motion.div>
          </div>
        )}

        {sessionSummary && !isStreaming && (
          <div key="summary-modal" className="fixed inset-0 z-[200] flex items-center justify-center bg-black/95 p-6">
            <motion.div 
              initial={{ y: 50, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
              className="glass-card w-full max-w-2xl p-10"
              style={{ backgroundColor: '#121216' }}
            >
              <h2 style={{ fontSize: '24px', fontWeight: '900', color: '#a78bfa', textTransform: 'uppercase', marginBottom: '1.5rem' }}>Mission Summary</h2>
              <div className="flex justify-between mb-8">
                <div>
                  <p style={{ fontSize: '10px', color: '#64748b' }}>Peak Load</p>
                  <p style={{ fontSize: '24px', fontWeight: '900' }}>{sessionSummary.session_summary?.peak_load}%</p>
                </div>
                <div>
                  <p style={{ fontSize: '10px', color: '#64748b' }}>Avg Load</p>
                  <p style={{ fontSize: '24px', fontWeight: '900' }}>{sessionSummary.session_summary?.mean_load}%</p>
                </div>
              </div>
              <button onClick={() => stopSession(null)} className="neo-button w-full">Dismiss</button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
