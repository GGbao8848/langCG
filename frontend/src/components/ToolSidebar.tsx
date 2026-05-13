import React from "react";
import { Search, Wrench } from "lucide-react";

interface ToolSidebarProps {
  isOpen: boolean;
  toolSearchQuery: string;
  setToolSearchQuery: (query: string) => void;
  filteredTools: any[];
}

const formatType = (prop: any): string => {
  if (!prop) return "unknown";
  if (prop.type) {
    if (prop.type === "array" && prop.items?.type) {
      return `${prop.items.type}[]`;
    }
    return String(prop.type);
  }
  if (Array.isArray(prop.anyOf)) {
    return prop.anyOf
      .map((item: any) => {
        if (item.type === "array" && item.items?.type) return `${item.items.type}[]`;
        return item.type ?? item.$ref?.split("/").pop() ?? "object";
      })
      .join(" | ");
  }
  if (prop.$ref) return prop.$ref.split("/").pop() ?? "object";
  return "object";
};

const formatDefault = (value: any): string => {
  if (value === undefined) return "";
  if (value === null) return "null";
  if (typeof value === "string") return value === "" ? "\"\"" : value;
  return JSON.stringify(value);
};

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
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wide font-mono mb-1 break-all">{tool.name}</h3>
              <p className="text-xs text-slate-600 mb-3 leading-relaxed break-words">{tool.description}</p>
              
              {tool.parameters.properties && Object.keys(tool.parameters.properties).length > 0 ? (
                <div className="space-y-2">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">参数</div>
                  {Object.entries(tool.parameters.properties).map(([key, prop]: [string, any]) => (
                    <div key={key} className="border-t border-slate-200 pt-2 first:border-t-0 first:pt-0">
                      <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                        <span className="min-w-0 max-w-full break-all font-mono font-semibold text-indigo-700">{key}</span>
                        <span className="rounded bg-slate-200 px-1.5 py-0.5 font-mono text-[10px] text-slate-700">{formatType(prop)}</span>
                        {tool.parameters.required?.includes(key) && (
                          <span className="rounded bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-600">必填</span>
                        )}
                        {prop.default !== undefined && (
                          <span className="min-w-0 max-w-full break-all rounded bg-emerald-50 px-1.5 py-0.5 font-mono text-[10px] text-emerald-700">
                            default={formatDefault(prop.default)}
                          </span>
                        )}
                      </div>
                      {prop.description && (
                        <span className="block text-slate-500 mt-1 whitespace-normal break-words text-xs leading-relaxed">{prop.description}</span>
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
