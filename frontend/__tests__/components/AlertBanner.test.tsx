import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import AlertBanner from '../../components/AlertBanner';

describe('AlertBanner', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it('does not render when alerts is null or empty', () => {
    const { container: containerNull } = render(<AlertBanner alerts={null} onDismiss={() => {}} />);
    expect(containerNull.firstChild).toBeNull();

    const { container: containerEmpty } = render(<AlertBanner alerts={[]} onDismiss={() => {}} />);
    expect(containerEmpty.firstChild).toBeNull();
  });

  it('renders the alerts list when populated', () => {
    const alerts = ['Patient exhibiting PVCs', 'Signal noise detected'];
    render(<AlertBanner alerts={alerts} onDismiss={() => {}} />);
    
    expect(screen.getByText('Clinical Anomaly Detected')).toBeInTheDocument();
    expect(screen.getByText('Patient exhibiting PVCs')).toBeInTheDocument();
    expect(screen.getByText('Signal noise detected')).toBeInTheDocument();
  });

  it('calls onDismiss when the dismiss button is clicked', () => {
    const onDismissMock = jest.fn();
    render(<AlertBanner alerts={['Test Alert']} onDismiss={onDismissMock} />);
    
    const dismissButton = screen.getByRole('button');
    fireEvent.click(dismissButton);
    
    expect(onDismissMock).toHaveBeenCalledTimes(1);
  });

  it('auto-dismisses after 8 seconds', () => {
    const onDismissMock = jest.fn();
    render(<AlertBanner alerts={['Test Alert']} onDismiss={onDismissMock} />);
    
    expect(onDismissMock).not.toHaveBeenCalled();
    
    act(() => {
      jest.advanceTimersByTime(7999);
    });
    expect(onDismissMock).not.toHaveBeenCalled();
    
    act(() => {
      jest.advanceTimersByTime(1);
    });
    expect(onDismissMock).toHaveBeenCalledTimes(1);
  });
});
