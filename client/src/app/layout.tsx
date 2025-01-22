import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Inter } from "next/font/google"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Location Guesser",
  description: "Geolocation guesser",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
    <head>
        <link rel="stylesheet" href="https://unpkg.com/react-globe.gl/dist/react-globe.gl.css"/>
    </head>
    <body
        className={`${inter.className} h-full`}
      >
        {children}
      </body>
    </html>
  );
}
