import React from 'react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { useStore } from '../../store/useStore';

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#0b0b0f]/95 backdrop-blur-xl border border-white/10 p-3 rounded-lg shadow-2xl">
        <p className="text-[8px] font-black tracking-widest text-slate-500 uppercase mb-2">Neural Window Data</p>
        <div className="flex flex-col gap-1.5 font-mono">
          <div className="flex items-center justify-between gap-6">
            <span className="text-[9px] font-black text-[#a78bfa] uppercase">Cognitive Load</span>
            <span className="text-xs font-black text-white">{payload[0]?.value?.toFixed(0)}%</span>
          </div>
          {payload[1] && (
            <div className="flex items-center justify-between gap-6">
              <span className="text-[9px] font-black text-[#fb7185] uppercase">Fatigue</span>
              <span className="text-xs font-black text-white">{payload[1]?.value?.toFixed(0)}%</span>
            </div>
          )}
          {payload[2] && (
            <div className="flex items-center justify-between gap-6">
              <span className="text-[9px] font-black text-[#10b981] uppercase">Attention</span>
              <span className="text-xs font-black text-white">{payload[2]?.value?.toFixed(0)}%</span>
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

export default function TemporalLoadChart() {
  const chartData = useStore((state) => state.chartData);

  return (
    <div className="w-full h-[220px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 15, right: 10, left: -22, bottom: 5 }}>
          <defs>
            <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.25}/>
              <stop offset="95%" stopColor="#a78bfa" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorFatigue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#fb7185" stopOpacity={0.15}/>
              <stop offset="95%" stopColor="#fb7185" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorAttention" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.15}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
            </linearGradient>
          </defs>
          
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.02)" vertical={false} />
          
          <XAxis 
            dataKey="time" 
            hide={true}
          />
          
          <YAxis 
            domain={[0, 100]} 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 8, fill: '#475569', fontWeight: 900 }}
            tickFormatter={(val) => `${val}%`}
            ticks={[0, 25, 50, 75, 100]}
          />
          
          <Tooltip content={<CustomTooltip />} />
          
          <ReferenceLine y={80} stroke="rgba(251, 113, 133, 0.4)" strokeDasharray="3 3" />
          <ReferenceLine y={65} stroke="rgba(234, 179, 8, 0.3)" strokeDasharray="3 3" />

          {/* Cognitive Load */}
          <Area 
            type="monotone" 
            dataKey="score" 
            stroke="#a78bfa" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#colorLoad)" 
            isAnimationActive={false}
          />
          
          {/* Fatigue */}
          <Area 
            type="monotone" 
            dataKey="fatigue" 
            stroke="#fb7185" 
            strokeWidth={1.5}
            fillOpacity={1} 
            fill="url(#colorFatigue)" 
            isAnimationActive={false}
          />

          {/* Attention */}
          <Area 
            type="monotone" 
            dataKey="attention" 
            stroke="#10b981" 
            strokeWidth={1.5}
            fillOpacity={1} 
            fill="url(#colorAttention)" 
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

