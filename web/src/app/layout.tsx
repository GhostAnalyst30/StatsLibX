import type { Metadata } from "next";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import { Toaster } from "sonner";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://statslibx.vercel.app"),
  title: {
    default: "StatsLibX — Statistical Analysis Library for Python",
    template: "%s — StatsLibX",
  },
  description:
    "Descriptive, inferential and computational statistics for Python: pandas/polars support, time series, distribution fitting, imputation, automatic EDA — plus an online playground that runs the real library in your browser.",
  keywords: [
    "statistics",
    "python",
    "data science",
    "descriptive statistics",
    "inferential statistics",
    "computational statistics",
    "time series",
    "eda",
    "imputation",
    "bootstrap",
  ],
  authors: [{ name: "Emmanuel Ascendra" }],
  openGraph: {
    type: "website",
    url: "https://statslibx.vercel.app",
    siteName: "StatsLibX",
    title: "StatsLibX — Statistical Analysis Library for Python",
    description:
      "Descriptive, inferential and computational statistics with time series, EDA and imputation — runs in your browser on the online playground.",
    images: [{ url: "/icons/favicon.svg", width: 512, height: 512, alt: "StatsLibX" }],
  },
  twitter: {
    card: "summary",
    title: "StatsLibX — Statistical Analysis Library",
    description:
      "Statistical analysis for Python: pandas/polars, time series, EDA, imputation. Try it in the browser playground.",
  },
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/icons/favicon.svg", type: "image/svg+xml" },
    ],
    apple: [{ url: "/icons/favicon.svg", type: "image/svg+xml" }],
    shortcut: ["/favicon.ico"],
  },
  manifest: "/site.webmanifest",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1">{children}</main>
        <Footer />
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: "#111118",
              border: "1px solid rgba(255,255,255,0.07)",
              color: "#e8e8f0",
              fontFamily: "DM Sans, sans-serif",
            },
          }}
        />
      </body>
    </html>
  );
}
