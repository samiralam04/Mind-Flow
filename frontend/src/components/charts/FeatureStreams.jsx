import React from 'react';
import { useStore } from '../../store/useStore';
import { Activity } from 'lucide-react';

function Sparkline({ history, color, width = 100, height = 24 }) {
  if (!history || history.length < 2) {
    return (
      <svg width={width} height={height} className="opacity-30">
        <line x1="0" y1={height/2} x2={width} y2={height/2} stroke={color} strokeWidth="1" strokeDasharray="2,2" />
      </svg>
    );
  }

  const min = Math.min(...history);
  const max = Math.max(...history);
  const range = max - min || 1;

  const points = history.map((val, index) => {
    const x = (index / (history.length - 1)) * width;
    const y = height - 2 - ((val - min) / range) * (height - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = `M ${points.join(' L ')}`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="transition-all duration-300"
      />
    </svg>
  );
}

function TelemetryTile({ label, value, metricKey, isStability = false, format = "0.00", colorClass = "text-cyan-400", strokeColor = "#22d3ee" }) {
  const chartData = useStore((state) => state.chartData);
  const liveMetrics = useStore((state) => state.liveMetrics);

  // Extract current value
  let raw = null;
  if (isStability) {
    const hp = Math.abs(Number(liveMetrics.head_pitch || 0));
    const hy = Math.abs(Number(liveMetrics.head_yaw || 0));
    const hr = Math.abs(Number(liveMetrics.head_roll || 0));
    raw = Math.max(50, Math.min(100, Math.round(100 - (hp + hy + hr) * 2.5)));
  } else {
    const val = liveMetrics[metricKey];
    raw = (val === "NaN" || val == null) ? null : Number(val);
  }

  const display = raw == null ? "—" : raw.toFixed((format.split('.')[1] || '').length);

  // Extract history
  const history = chartData.map((pt) => {
    if (isStability) {
      const hp = Math.abs(Number(pt.metrics?.head_pitch || 0));
      const hy = Math.abs(Number(pt.metrics?.head_yaw || 0));
      const hr = Math.abs(Number(pt.metrics?.head_roll || 0));
      return Math.max(50, Math.min(100, Math.round(100 - (hp + hy + hr) * 2.5)));
    }
    const val = pt.metrics?.[metricKey];
    return (val === "NaN" || val == null) ? 0 : Number(val);
  }).slice(-30); // Grab last 30 data points for high resolution

  return (
    <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#0e0e12] border border-white/[0.04] hover:border-white/[0.08] transition-all duration-300">
      <div className="flex flex-col gap-0.5">
        <span className="text-[8px] font-black tracking-widest text-slate-500 uppercase">{label}</span>
        <span className="text-xs font-mono font-black text-white">{display}</span>
      </div>
      <div className="flex items-center">
        <Sparkline history={history} color={strokeColor} width={200} height={26} />
      </div>
    </div>
  );
}

export default function FeatureStreams() {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between pb-1.5 border-b border-white/[0.05]">
        <div className="flex items-center gap-2">
          <Activity size={12} className="text-[#a78bfa]" />
          <h3 className="text-[10px] font-black tracking-[0.2em] uppercase text-slate-300">Real-Time Physiology</h3>
        </div>
      </div>
      
      <div className="flex flex-col gap-2">
        <TelemetryTile label="Eye Aspect Ratio" metricKey="ear" format="0.00" strokeColor="#3b82f6" />
        <TelemetryTile label="Eye Openness" metricKey="eye_openness" format="0.00" strokeColor="#3b82f6" />
        <TelemetryTile label="Brow Tension" metricKey="eyebrow_tension" format="0.00" strokeColor="#a78bfa" />
        <TelemetryTile label="Lumen Density" metricKey="light_intensity" format="0.0" strokeColor="#eab308" />
        <TelemetryTile label="Pitch" metricKey="head_pitch" format="+0.0°" strokeColor="#06b6d4" />
        <TelemetryTile label="Yaw" metricKey="head_yaw" format="+0.0°" strokeColor="#06b6d4" />
        <TelemetryTile label="Roll" metricKey="head_roll" format="+0.0°" strokeColor="#06b6d4" />
        <TelemetryTile label="Head Stability" isStability={true} format="0%" strokeColor="#10b981" />
      </div>
    </div>
  );
}

