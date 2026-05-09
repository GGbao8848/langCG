import React, { useState } from "react";
import { MessageSquare, Plus, MoreHorizontal, Pencil, Trash2, X, Check, Settings2 } from "lucide-react";
import { ChatSession } from "../types";

interface ChatSidebarProps {
  isOpen: boolean;
  sessions: ChatSession[];
  currentSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newName: string) => void;
  onLLMSettingsClick?: () => void;
  activeModelLabel?: string;
}

export const ChatSidebar: React.FC<ChatSidebarProps> = ({
  isOpen,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  onLLMSettingsClick,
  activeModelLabel,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [menuPos, setMenuPos] = useState<{ id: string, top: number, left: number } | null>(null);

  const startEditing = (id: string, currentName: string) => {
    setEditingId(id);
    setEditName(currentName);
    setMenuPos(null);
  };

  const handleRename = () => {
    if (editingId && editName.trim()) {
      onRenameSession(editingId, editName.trim());
    }
    setEditingId(null);
  };

  const sortedSessions = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <div style={{ width: isOpen ? '280px' : '0' }} className="bg-slate-50 border-r border-slate-200 transition-all duration-300 flex flex-col shrink-0 overflow-hidden relative z-30">
      <div className="w-[280px] flex-1 flex flex-col min-h-0 absolute inset-0">
        <div className="pt-5 pb-2 px-4 shrink-0 flex justify-center">
          <button
            onClick={onNewSession}
            className="w-[90%] flex items-center justify-center gap-2 py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl transition-colors text-sm font-medium shadow-sm shadow-indigo-200/50"
          >
            <Plus className="w-4 h-4 text-white" />
            新会话
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto scrollbar-thin px-3 pb-3 space-y-1">
          {sortedSessions.map((session) => (
            <div
              key={session.id}
              className={`group relative flex items-center p-2 rounded-lg cursor-pointer transition-colors ${
                currentSessionId === session.id
                  ? "bg-indigo-50 text-indigo-700"
                  : "hover:bg-slate-200 text-slate-700"
              }`}
              onClick={() => {
                if (editingId !== session.id) {
                  onSelectSession(session.id);
                }
              }}
            >
              <MessageSquare className={`w-4 h-4 mr-3 shrink-0 ${currentSessionId === session.id ? 'text-indigo-500' : 'text-slate-400'}`} />
              
              {editingId === session.id ? (
                <div className="flex-1 flex items-center min-w-0 mr-1 gap-1">
                  <input
                    type="text"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRename();
                      if (e.key === 'Escape') setEditingId(null);
                    }}
                    autoFocus
                    className="flex-1 min-w-0 bg-white border border-indigo-300 rounded px-1.5 py-0.5 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500 text-slate-800"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <button onClick={(e) => { e.stopPropagation(); handleRename(); }} className="text-emerald-600 hover:bg-emerald-100 p-1 rounded shrink-0">
                    <Check className="w-3 h-3" />
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); setEditingId(null); }} className="text-slate-500 hover:bg-slate-200 p-1 rounded shrink-0">
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ) : (
                <div className="flex-1 truncate text-sm select-none" title={session.name}>
                  {session.name}
                </div>
              )}

              {editingId !== session.id && (
                <div className="relative shrink-0 ml-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (menuPos?.id === session.id) {
                        setMenuPos(null);
                      } else {
                        const rect = e.currentTarget.getBoundingClientRect();
                        setMenuPos({
                          id: session.id,
                          top: rect.bottom + 4,
                          left: rect.right - 128, // w-32 is 128px
                        });
                      }
                    }}
                    className={`p-1 rounded-md transition-opacity ${menuPos?.id === session.id ? 'opacity-100 bg-slate-300' : 'opacity-0 group-hover:opacity-100 hover:bg-slate-300'}`}
                  >
                    <MoreHorizontal className="w-4 h-4 text-slate-500" />
                  </button>

                  {menuPos?.id === session.id && (
                    <>
                      <div className="fixed inset-0 z-40" onClick={(e) => { e.stopPropagation(); setMenuPos(null); }}></div>
                      <div 
                        className="fixed w-32 py-1 bg-white border border-slate-200 rounded-lg shadow-lg z-50 text-sm"
                        style={{ top: menuPos.top, left: menuPos.left }}
                      >
                        <button
                          className="w-full text-left px-3 py-1.5 text-slate-700 hover:bg-slate-50 flex items-center"
                          onClick={(e) => {
                            e.stopPropagation();
                            startEditing(session.id, session.name);
                          }}
                        >
                          <Pencil className="w-3.5 h-3.5 mr-2" /> 命名
                        </button>
                        <button
                          className="w-full text-left px-3 py-1.5 text-red-600 hover:bg-red-50 flex items-center"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSession(session.id);
                            setMenuPos(null);
                          }}
                        >
                          <Trash2 className="w-3.5 h-3.5 mr-2" /> 删除
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
        
        <div className="p-3 border-t border-slate-200 shrink-0">
          <button
            onClick={onLLMSettingsClick}
            className="w-full flex items-center p-2 rounded-xl transition-all duration-200 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/50 bg-white border border-slate-200 hover:border-indigo-200 hover:bg-indigo-50/50 text-slate-700"
          >
            <div className="shrink-0 flex items-center justify-center w-8 h-8 rounded-full mr-2.5 bg-slate-100 text-slate-500">
              <Settings2 className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0 text-left flex flex-col">
              <span className="text-sm font-medium truncate">当前模型</span>
              <span className="text-[10px] truncate text-slate-400">{activeModelLabel ?? "OpenRouter / Ollama"}</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};
