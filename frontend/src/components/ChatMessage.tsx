import React from "react";
import { Bot, ChevronRight, Loader2, User, Wrench } from "lucide-react";
import { UIMessage } from "../types";

export const ChatMessage: React.FC<{ msg: UIMessage; loggedInUser?: string | null }> = ({ msg, loggedInUser }) => {
  return (
    <div
      className={`flex flex-col w-full ${msg.role === "user" ? "items-end" : "items-start"}`}
    >
      <div
        className={`flex max-w-[85%] sm:max-w-5xl gap-3 ${
          msg.role === "user" ? "flex-row-reverse" : "flex-row"
        }`}
      >
        {/* Avatar */}
        <div
          className={`mt-1 shrink-0 flex items-center justify-center w-8 h-8 rounded-full shadow-sm ${
            msg.role === "user" ? "bg-indigo-600" : "bg-emerald-500"
          }`}
        >
          {msg.role === "user" ? (
            loggedInUser ? (
              <span className="text-white text-xs font-semibold tracking-wider">
                {loggedInUser.substring(0, 2).toUpperCase()}
              </span>
            ) : (
              <User className="w-5 h-5 text-white" />
            )
          ) : (
            <Bot className="w-5 h-5 text-white" />
          )}
        </div>

        {/* Content Bubble */}
        <div className="flex flex-col justify-start space-y-2 min-w-0">
          {/* Tool Invocations Box */}
          {msg.toolCalls && msg.toolCalls.length > 0 && (
            <div className="flex flex-col space-y-2 w-full min-w-0 sm:min-w-[280px]">
              {msg.toolCalls.map((tc) => {
                let statusConfig = {
                  border: "border-slate-200",
                  summaryBg: "bg-slate-50",
                  summaryHoverBg: "hover:bg-slate-100",
                  textColor: "text-slate-600",
                  iconColor: "text-slate-400",
                  text: "Success",
                  badgeTextColor: "text-emerald-600",
                };
                
                if (tc.status === "running") {
                  statusConfig = {
                    border: "border-amber-200",
                    summaryBg: "bg-amber-50",
                    summaryHoverBg: "hover:bg-amber-100",
                    textColor: "text-amber-700",
                    iconColor: "text-amber-500",
                    text: "Running...",
                    badgeTextColor: "text-amber-600",
                  };
                } else if (tc.status === "error") {
                  statusConfig = {
                    border: "border-red-200",
                    summaryBg: "bg-red-50",
                    summaryHoverBg: "hover:bg-red-100",
                    textColor: "text-red-700",
                    iconColor: "text-red-500",
                    text: "Failed",
                    badgeTextColor: "text-red-600",
                  };
                } else if (tc.status === "success" || tc.status === "done") {
                  statusConfig = {
                    border: "border-emerald-200",
                    summaryBg: "bg-emerald-50",
                    summaryHoverBg: "hover:bg-emerald-100",
                    textColor: "text-emerald-700",
                    iconColor: "text-emerald-500",
                    text: "Success",
                    badgeTextColor: "text-emerald-600",
                  };
                }

                return (
                <details key={tc.id} className={`bg-white border ${statusConfig.border} rounded-xl shadow-sm text-sm group transition-colors`}>
                  <summary className={`${statusConfig.summaryBg} px-3 py-2 font-medium ${statusConfig.textColor} flex items-center text-xs cursor-pointer select-none outline-none group-open:border-b group-open:${statusConfig.border} list-none [&::-webkit-details-marker]:hidden rounded-xl group-open:rounded-b-none transition-colors ${statusConfig.summaryHoverBg}`}>
                    <ChevronRight className={`w-3.5 h-3.5 mr-1 transition-transform group-open:rotate-90 ${statusConfig.iconColor}`} />
                    <Wrench className={`w-3.5 h-3.5 mr-1.5 ${statusConfig.iconColor}`} />
                    <span className="font-mono tracking-wide">{tc.name}</span>
                    <span className={`ml-auto font-medium flex items-center ${statusConfig.badgeTextColor} ${tc.status === 'running' ? 'animate-pulse' : ''}`}>
                      {tc.status === "running" && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
                      {statusConfig.text}
                    </span>
                  </summary>
                  <div className="p-0 overflow-hidden">
                    <div className="p-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-slate-700 text-xs uppercase tracking-wider font-mono">
                          {tc.name}
                        </span>
                      </div>
                      <div className="text-xs font-mono bg-slate-50 text-slate-500 p-2 rounded -mx-1 mt-2 mb-1 overflow-x-auto whitespace-pre-wrap word-break">
                        <span className="text-slate-400 select-none">Args:</span>{" "}
                        {JSON.stringify(tc.args)}
                      </div>
                      {tc.status !== "running" && tc.result && (
                        <div className="text-xs font-mono bg-emerald-50 text-emerald-700 p-2 rounded -mx-1 mt-1 border border-emerald-100 overflow-x-auto whitespace-pre-wrap word-break">
                          <span className="text-emerald-600 font-medium select-none">Result:</span>{" "}
                          {JSON.stringify(tc.result)}
                        </div>
                      )}
                    </div>
                  </div>
                </details>
                );
              })}
            </div>
          )}

          {/* Text Message */}
          {msg.text && (
            <div
              className={`px-4 py-2.5 rounded-2xl whitespace-pre-wrap relative shadow-sm ${
                msg.role === "user"
                  ? "bg-indigo-600 text-white rounded-tr-sm"
                  : "bg-white border border-slate-200 text-slate-800 rounded-tl-sm"
              }`}
            >
              {msg.text}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
