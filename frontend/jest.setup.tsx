import '@testing-library/jest-dom';
import React from 'react';

// Mock Recharts to avoid canvas errors in jsdom
jest.mock('recharts', () => {
  const OriginalModule = jest.requireActual('recharts');
  return {
    ...OriginalModule,
    ResponsiveContainer: ({ children }: any) => <div data-testid="ResponsiveContainer">{children}</div>,
    LineChart: ({ children }: any) => <div data-testid="LineChart">{children}</div>,
    Line: (props: any) => <div data-testid="Line" {...props} />,
    XAxis: (props: any) => <div data-testid="XAxis" {...props} />,
    YAxis: (props: any) => <div data-testid="YAxis" {...props} />,
    CartesianGrid: (props: any) => <div data-testid="CartesianGrid" {...props} />,
    Tooltip: (props: any) => <div data-testid="Tooltip" {...props} />,
    ReferenceDot: (props: any) => <div data-testid="ReferenceDot" {...props} />,
    ReferenceArea: (props: any) => <div data-testid="ReferenceArea" {...props} />,
  };
});
