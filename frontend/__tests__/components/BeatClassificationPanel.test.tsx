import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import BeatClassificationPanel from '../../components/BeatClassificationPanel';
import { BeatEvent } from '../../hooks/useECGStream';

jest.mock('../../components/BeatInspector', () => {
  return function MockBeatInspector({ beat, onClose }: { beat: any, onClose: any }) {
    if (!beat) return null;
    return (
      <div data-testid="mock-beat-inspector">
        <span>Mock Inspector for {beat.beat_type}</span>
        <button onClick={onClose} data-testid="close-inspector">Close</button>
      </div>
    );
  };
});

describe('BeatClassificationPanel', () => {
  const createMockBeat = (overrides?: Partial<BeatEvent>): BeatEvent => ({
    timestamp: Date.now() / 1000,
    bpm: 72,
    beat_type: 'N',
    confidence: 0.9,
    rhythm_class: 'Regular',
    anomaly_score: 0.1,
    raw_window: new Array(360).fill(0),
    alert: null,
    ...overrides,
  });

  it('renders "No beats yet" state when history is empty', () => {
    render(<BeatClassificationPanel history={[]} />);
    expect(screen.getByText('No beats yet')).toBeInTheDocument();
  });

  it('renders correct beat type badge color for each class', () => {
    const classes = ['N', 'V', 'A', 'L', 'R'];
    const history = classes.map((type, i) => createMockBeat({ beat_type: type, timestamp: i }));
    render(<BeatClassificationPanel history={history} />);
    
    const nBadge = screen.getByText('Normal').closest('span');
    expect(nBadge).toHaveClass('bg-green-500/20');
    
    const vBadge = screen.getByText('PVC').closest('span');
    expect(vBadge).toHaveClass('bg-red-500/20');
    
    const aBadge = screen.getByText('APB').closest('span');
    expect(aBadge).toHaveClass('bg-orange-500/20');
    
    const lBadge = screen.getByText('LBBB').closest('span');
    expect(lBadge).toHaveClass('bg-blue-500/20');
    
    const rBadge = screen.getByText('RBBB').closest('span');
    expect(rBadge).toHaveClass('bg-purple-500/20');
  });

  it('truncates the list to the last 10 items if history length is > 10', () => {
    const history = Array.from({ length: 15 }, (_, i) => createMockBeat({ timestamp: i }));
    render(<BeatClassificationPanel history={history} />);
    
    const beatItems = screen.getAllByRole('button');
    expect(beatItems).toHaveLength(10);
  });

  it('clicking a beat item calls openInspector/modal state', () => {
    const history = [createMockBeat({ beat_type: 'V', timestamp: 12345 })];
    render(<BeatClassificationPanel history={history} />);
    
    expect(screen.queryByTestId('mock-beat-inspector')).not.toBeInTheDocument();
    
    const beatButton = screen.getAllByRole('button')[0];
    fireEvent.click(beatButton);
    
    expect(screen.getByTestId('mock-beat-inspector')).toBeInTheDocument();
    expect(screen.getByText('Mock Inspector for V')).toBeInTheDocument();
    
    fireEvent.click(screen.getByTestId('close-inspector'));
    expect(screen.queryByTestId('mock-beat-inspector')).not.toBeInTheDocument();
  });
});
