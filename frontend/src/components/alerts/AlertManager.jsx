import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useStore } from '../../store/useStore';

export default function AlertManager() {
  const alerts = useStore((state) => state.alerts);
  const removeAlert = useStore((state) => state.removeAlert);

  // Auto-remove alerts after 4 seconds
  React.useEffect(() => {
    alerts.forEach((alert) => {
      const timer = setTimeout(() => {
        removeAlert(alert.id);
      }, 4000);
      return () => clearTimeout(timer);
    });
  }, [alerts, removeAlert]);

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 pointer-events-none">
      <AnimatePresence>
        {alerts.map((alert) => (
          <motion.div
            key={alert.id}
            initial={{ opacity: 0, x: 50, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
            className="flex items-center gap-3 px-4 py-3 rounded-xl border bg-gray-900/90 backdrop-blur-md shadow-2xl pointer-events-auto"
            style={{ borderColor: alert.color }}
          >
            <span className="text-2xl">{alert.emoji}</span>
            <div className="flex flex-col">
              <span className="text-[10px] text-gray-400 font-mono uppercase tracking-wider">State Transition</span>
              <span className="text-sm font-bold" style={{ color: alert.color }}>{alert.label}</span>
            </div>
            <button 
              onClick={() => removeAlert(alert.id)}
              className="ml-4 text-gray-500 hover:text-white transition"
            >
              ×
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
