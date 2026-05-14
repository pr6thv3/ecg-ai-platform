import '@testing-library/jest-dom';
import React from 'react';

// Mock Recharts to avoid canvas errors in jsdom
jest.mock('recharts', () => {
  const OriginalModule = jest.requireActual('recharts');
  return {
    ...OriginalModule,
    ResponsiveContainer: ({ children }: any) => <div data-testid="ResponsiveContainer">{children}</div>,
    LineChart: ({ children }: any) => <div data-testid="LineChart">{children}</div>,
    Line: () => <div data-testid="Line" />,
    XAxis: () => <div data-testid="XAxis" />,
    YAxis: () => <div data-testid="YAxis" />,
    CartesianGrid: () => <div data-testid="CartesianGrid" />,
    Tooltip: () => <div data-testid="Tooltip" />,
    ReferenceDot: () => <div data-testid="ReferenceDot" />,
    ReferenceArea: () => <div data-testid="ReferenceArea" />,
  };
});
