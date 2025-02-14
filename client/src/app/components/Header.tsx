"use client"

import Link from "next/link"
import { MapPin } from "lucide-react"

export default function Header() {

    return (
        <header className="bg-gray-800 text-white relative z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                <div className="flex items-center justify-between">
                    <Link href="/" className="flex items-center space-x-2">
                        <MapPin className="h-8 w-8 text-blue-400" />
                        <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">LocationGuesser</span>
                    </Link>
                    <nav className="">
                        <ul className="flex space-x-6">
                            <li>
                                <Link href="/" className="text-gray-300 hover:text-white transition-colors hidden md:block">
                                    Home
                                </Link>
                            </li>
                            <li>
                                <button>
                                    <Link href="/predict" className="font-semibold text-blue-400">Try It Now</Link>
                                </button>
                            </li>
                        </ul>
                    </nav>
                </div>
            </div>
        </header>
    )
}

