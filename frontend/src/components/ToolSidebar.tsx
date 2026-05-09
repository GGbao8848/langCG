import React from "react";
import { Search, Wrench } from "lucide-react";

interface ToolSidebarProps {
  isOpen: boolean;
  toolSearchQuery: string;
  setToolSearchQuery: (query: string) => void;
  filteredTools: any[];
}

export const ToolSidebar: React.FC<ToolSidebarProps> = ({
  isOpen,
  toolSearchQuery,
  setToolSearchQuery,
  filteredTools,
}) => {
  return (
    <div style={{ width: isOpen ? '320px' : '0' }} className="bg-white border-l border-slate-200 transition-all duration-300 flex flex-col shrink-0 overflow-hidden relative">
      <div className="w-80 flex-1 flex flex-col min-h-0 absolute inset-0">
        <div className="p-4 border-b border-slate-200 shrink-0 shadow-sm flex flex-col gap-3">
          <h2 className="text-sm font-semibold text-slate-800 flex items-center">
            <Wrench className="w-4 h-4 mr-2 text-indigo-500" />
            Available Tools
          </h2>
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
            <input
              type="text"
              placeholder="搜索工具..."
              className="w-full bg-slate-50 border border-slate-200 rounded-lg py-2 pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-400 transition-colors"
              value={toolSearchQuery}
              onChange={(e) => setToolSearchQuery(e.target.value)}
            />
          </div>
        </div>
        <div className="p-4 overflow-y-auto scrollbar-thin flex-1 space-y-4">
          {filteredTools.map((tool) => (
            <div key={tool.name} className="border border-slate-200 rounded-lg p-4 bg-slate-50 shadow-sm">
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wide font-mono mb-1">{tool.name}</h3>
              <p className="text-xs text-slate-600 mb-3">{tool.description}</p>
              
              {tool.parameters.properties && Object.keys(tool.parameters.properties).length > 0 ? (
                <div className="space-y-1.5">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Parameters:</div>
                  {Object.entries(tool.parameters.properties).map(([key, prop]: [string, any]) => (
                    <div key={key} className="text-[11px] bg-white border border-slate-200 rounded-md p-2 font-mono shadow-sm">
                      <div className="flex items-center mb-1">
                        <span className="text-indigo-600 font-semibold">{key}</span>
                        <span className="text-slate-400 mx-1.5">:</span>
                        <span className="text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded text-[10px]">{prop.type}</span>
                        {tool.parameters.required?.includes(key) && (
                          <span className="ml-2 text-red-500 text-[10px] font-sans font-semibold">required</span>
                        )}
                      </div>
                      {prop.description && (
                        <span className="block text-slate-500 mt-1 whitespace-normal text-xs font-sans">{prop.description}</span>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-[11px] font-medium text-slate-400 bg-white border border-slate-200 rounded inline-block px-2 py-1 italic">
                  No parameters
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
