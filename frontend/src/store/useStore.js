import { create } from 'zustand';

const MAX_HISTORY = 150; // 30 seconds at 5 FPS

export const useStore = create((set, get) => ({
  // WebSocket Status
  isConnected: false,
  isStreaming: false,
  sessionId: null,
  mlReady: false,
  
  // Real-time Current Values
  cogLoad: 35,
  mlData: {},
  liveMetrics: {},
  sessionSummary: null,
  framesProcessed: 0,
  sessionStartTime: null,
  
  // Temporal History (for graphs)
  chartData: [], 
  
  // Alerts
  alerts: [],
  
  // Actions
  setConnectionStatus: (status) => set({ isConnected: status }),
  startSession: (id, ready) => set({ 
    isStreaming: true, 
    sessionId: id, 
    mlReady: ready, 
    chartData: [], 
    alerts: [],
    sessionSummary: null,
    framesProcessed: 0,
    sessionStartTime: Date.now(),
    cogLoad: 35
  }),
  stopSession: (summary) => set({ isStreaming: false, sessionId: null, sessionSummary: summary }),
  
  addFrameData: (data) => set((state) => {
    const now = Date.now();
    const score = Number(data.score);
    const fatigueVal = data.ml?.fatigue != null ? data.ml.fatigue * 100 : 0;
    
    // Dynamic Attention Calculation
    // Attention is high when focused, decreases during fatigue, extreme head movements or distraction.
    let attentionVal = 100 - score;
    if (data.ml?.state === 'distracted') attentionVal = 25;
    else if (data.ml?.state === 'fatigued') attentionVal = 35;
    else if (data.ml?.state === 'overloaded') attentionVal = 40;
    
    // Cap attention boundaries
    attentionVal = Math.max(15, Math.min(95, attentionVal));

    const newPoint = {
      time: now,
      score: score,
      raw_load: data.ml?.raw_load != null ? data.ml.raw_load * 100 : score,
      fatigue: fatigueVal,
      attention: attentionVal,
      state: data.ml?.state || 'unknown',
      metrics: data.metrics || {}
    };
    
    // Downsample: add point every 200ms (5 FPS) for Recharts performance
    let newHistory = state.chartData;
    const lastPoint = newHistory[newHistory.length - 1];
    if (!lastPoint || now - lastPoint.time >= 200) {
      newHistory = [...newHistory, newPoint].slice(-MAX_HISTORY);
    }
    
    // Alert System
    let newAlerts = [...state.alerts];
    if (data.ml?.state && data.ml.state !== 'unknown') {
      const lastState = state.mlData?.state;
      if (data.ml.state !== lastState && lastState) {
        newAlerts.push({
          id: Date.now(),
          type: 'state_change',
          state: data.ml.state,
          label: data.ml.state_label,
          color: data.ml.state_color,
          emoji: data.ml.state_emoji
        });
        if (newAlerts.length > 3) newAlerts.shift(); // keep max 3 toast alerts
      }
    }

    return {
      cogLoad: score,
      mlData: data.ml || {},
      liveMetrics: data.metrics || {},
      chartData: newHistory,
      alerts: newAlerts,
      framesProcessed: state.framesProcessed + 1
    };
  }),

  removeAlert: (id) => set((state) => ({
    alerts: state.alerts.filter(a => a.id !== id)
  }))
}));

