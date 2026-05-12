import { render, screen } from '@testing-library/react';
import Dashboard from '../app/dashboard/page';
import ECGWaveformChart from '../components/ECGWaveformChart';
import ErrorBoundary from '../components/ErrorBoundary';

// 1. Mock Recharts to avoid ResizeObserver/DOM rendering errors in Jest node environment
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  LineChart: () => <div>LineChart</div>,
  Line: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  Tooltip: () => <div />,
}));

test('Dashboard page renders without crashing', () => {
  render(<Dashboard />);
  expect(screen.getByText(/ECG AI Platform/i)).toBeInTheDocument();
});

test('ECGWaveformChart renders with empty data without throwing', () => {
  render(<ECGWaveformChart data={[]} />);
  expect(screen.getByText('LineChart')).toBeInTheDocument();
});

test('ErrorBoundary catches a child component error and shows fallback UI', () => {
  // Suppress expected console.error from Error boundary
  const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  
  const ThrowError = () => { throw new Error('Simulated Crash'); };
  render(
    <ErrorBoundary>
      <ThrowError />
    </ErrorBoundary>
  );
  expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
  
  consoleSpy.mockRestore();
});
