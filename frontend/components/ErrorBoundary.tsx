"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return (
        <div className="p-4 m-4 bg-red-900/20 border border-red-500 rounded-lg text-white">
          <h2 className="text-xl font-bold mb-2 text-red-400">Component Error</h2>
          <p className="text-sm text-gray-300">
            Something went wrong while rendering this section.
          </p>
          <pre className="mt-2 text-xs text-red-300 bg-black/50 p-2 rounded overflow-auto">
            {this.state.error?.message}
          </pre>
        </div>
      );
    }

    return this.props.children;
  }
}
