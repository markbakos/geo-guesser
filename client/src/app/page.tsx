"use client"

import Header from "./components/Header"
import dynamic from "next/dynamic"
import { Github } from "lucide-react"
import Link from "next/link"

const AnimatedGlobe = dynamic(() => import("./components/Globe"), { ssr: false })

export default function Home() {
    return (
        <div className="min-h-screen flex flex-col bg-gray-900 text-white overflow-x-hidden">
            <Header />
            <main className="flex-grow">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-24">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                        <div className="space-y-6">
                            <h1 className="text-4xl font-extrabold sm:text-5xl md:text-6xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
                                Discover Where Your Images Were Taken
                            </h1>
                            <div className="flex flex-col sm:flex-row gap-4">
                                <Link href="/predict" >
                                    <button className="bg-blue-600 hover:bg-blue-700 w-32 h-10 rounded-md">
                                        Get Started
                                    </button>
                                </Link>
                                <Link href="https://github.com/markbakos/geo-guesser" target="_blank">
                                    <button
                                        className="gap-2 flex items-center justify-center w-48 h-10 border border-white rounded-md hover:text-black hover:bg-white transition">
                                        <Github className="w-5 h-5"/>
                                        View on GitHub
                                    </button>
                                </Link>
                            </div>
                        </div>
                        <div className="w-full lg:w-1/2 flex justify-center items-center mt-8 lg:mt-0">
                            <div
                                className="w-full max-w-[300px] sm:max-w-[400px] md:max-w-[500px] aspect-square relative">
                                <div
                                    className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full blur-3xl"></div>
                                <AnimatedGlobe/>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    )
}

