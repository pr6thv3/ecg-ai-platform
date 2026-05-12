import React from 'react';
import { render, screen, act } from '@testing-library/react';
import MetricsBar from '../../components/MetricsBar';

describe('MetricsBar', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it('renders default empty state when bpm is 0', () => {
    render(<MetricsBar bpm={0} rhythm="Regular" anomaly={0} status="disconnected" />);
    
    expect(screen.getByText('--')).toBeInTheDocument();
    expect(screen.getByText('Analyzing...')).toBeInTheDocument();
    expect(screen.getByText('0.0% Divergence')).toBeInTheDocument();
    expect(screen.getByText('00:00')).toBeInTheDocument();
  });

  it('renders correct bpm and rhythm when active', () => {
    render(<MetricsBar bpm={75} rhythm="Regular" anomaly={0.1} status="connected" />);
    
    expect(screen.getByText('75')).toBeInTheDocument();
    expect(screen.getByText('Regular')).toBeInTheDocument();
    expect(screen.getByText('10.0% Divergence')).toBeInTheDocument();
  });

  it('renders correct rhythm colors', () => {
    const { rerender } = render(<MetricsBar bpm={75} rhythm="Regular" anomaly={0} status="connected" />);
    expect(screen.getByText('Regular')).toHaveClass('text-green-400');

    rerender(<MetricsBar bpm={75} rhythm="Irregular" anomaly={0} status="connected" />);
    expect(screen.getByText('Irregular')).toHaveClass('text-orange-400');

    rerender(<MetricsBar bpm={120} rhythm="Tachycardia" anomaly={0} status="connected" />);
    expect(screen.getByText('Tachycardia')).toHaveClass('text-red-400');
  });

  it('anomaly score renders correct threshold colors', () => {
    const { container, rerender } = render(<MetricsBar bpm={75} rhythm="Regular" anomaly={0.2} status="connected" />);
    
    // The bar is a div inside the Anomaly Score section.
    // It should have bg-blue-500 because anomaly is 0.2 <= 0.4
    let bar = container.querySelector('.bg-blue-500');
    expect(bar).toBeInTheDocument();
    expect(bar).toHaveStyle({ width: '20%' });

    rerender(<MetricsBar bpm={75} rhythm="Regular" anomaly={0.5} status="connected" />);
    bar = container.querySelector('.bg-red-500');
    expect(bar).toBeInTheDocument();
    expect(bar).toHaveStyle({ width: '50%' });
  });

  it('session duration timer ticks up every second while connected', () => {
    render(<MetricsBar bpm={75} rhythm="Regular" anomaly={0} status="connected" />);
    
    expect(screen.getByText('00:00')).toBeInTheDocument();
    
    act(() => {
      jest.advanceTimersByTime(1000);
    });
    expect(screen.getByText('00:01')).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(59000); // jump 59 seconds to make it 1 minute
    });
    expect(screen.getByText('01:00')).toBeInTheDocument();
  });

  it('timer does not tick if status is disconnected', () => {
    render(<MetricsBar bpm={0} rhythm="Regular" anomaly={0} status="disconnected" />);
    
    expect(screen.getByText('00:00')).toBeInTheDocument();
    
    act(() => {
      jest.advanceTimersByTime(5000);
    });
    expect(screen.getByText('00:00')).toBeInTheDocument(); // Still 0
  });
});
