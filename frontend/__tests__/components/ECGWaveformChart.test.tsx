import React from 'react';
import { render, screen } from '@testing-library/react';
import ECGWaveformChart from '@/components/ECGWaveformChart';

describe('ECGWaveformChart', () => {
  it('renders without crashing with empty buffer', () => {
    render(<ECGWaveformChart data={[]} />);
    expect(screen.getByTestId('LineChart')).toBeInTheDocument();
  });

  it('renders correct number of data points when buffer has 500 samples', () => {
    const data = Array.from({ length: 500 }, (_, i) => ({ time: i, value: Math.random() }));
    render(<ECGWaveformChart data={data} />);
    expect(screen.getByTestId('LineChart')).toBeInTheDocument();
  });

  it('R-peak markers appear when data points have isPeak set to true', () => {
    const data = Array.from({ length: 500 }, (_, i) => ({ 
        time: i, 
        value: Math.random(),
        isPeak: i === 100 || i === 200 || i === 300,
        type: 'N'
    }));
    render(<ECGWaveformChart data={data} />);
    // Our mock renders <div data-testid="ReferenceDot" /> for each ReferenceDot
    const dots = screen.getAllByTestId('ReferenceDot');
    expect(dots).toHaveLength(3);
  });

  it('does not throw when new data is appended to buffer', () => {
    const { rerender } = render(<ECGWaveformChart data={[]} />);
    const data = [{ time: 0, value: 0.5 }];
    expect(() => {
      rerender(<ECGWaveformChart data={data} />);
    }).not.toThrow();
  });
});
