import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'ECG AI Monitor — Real-Time Arrhythmia Analysis',
  description: 'Real-time ECG arrhythmia classification dashboard powered by 1D CNN inference via ONNX Runtime',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
