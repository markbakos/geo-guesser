"use client"

import { useState } from "react"
import Link from "next/link"
import { MapPin, Menu, X } from "lucide-react"
import ContactPopup from "@/app/components/ContactPopup"

export default function Header() {
    const [isContactOpen, setIsContactOpen] = useState(false)
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

    const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen)

    return (
        <header className="bg-white shadow-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                <div className="flex items-center justify-between">
                    <Link href="/">
                        <div className="flex items-center cursor-pointer">
                            <MapPin className="h-8 w-8 text-blue-600" />
                            <span className="ml-2 text-xl font-bold text-gray-900 select-none">LocationGuesser</span>
                        </div>
                    </Link>
                    <nav className="hidden md:block">
                        <ul className="flex space-x-4">
                            <li>
                                <Link href="/" className="text-gray-600 hover:text-gray-900">
                                    Home
                                </Link>
                            </li>
                            <li>
                                <button
                                    onClick={() => setIsContactOpen(true)}
                                    className="text-gray-600 hover:text-blue-600 transition-colors"
                                >
                                    Contact
                                </button>
                            </li>
                            <li>
                                <Link href="/predict" className="text-blue-600 hover:text-blue-700 font-medium">
                                    Try It Now
                                </Link>
                            </li>
                        </ul>
                    </nav>
                    <div className="md:hidden">
                        <button onClick={toggleMobileMenu} className="text-gray-600 hover:text-gray-900">
                            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                        </button>
                    </div>
                </div>
            </div>
            {isMobileMenuOpen && (
                <div className="md:hidden">
                    <nav className="px-2 pt-2 pb-4 space-y-1 sm:px-3">
                        <Link
                            href="/"
                            className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-50"
                        >
                            Home
                        </Link>
                        <button
                            onClick={() => {
                                setIsContactOpen(true)
                                setIsMobileMenuOpen(false)
                            }}
                            className="block w-full text-left px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-50"
                        >
                            Contact
                        </button>
                        <Link
                            href="/predict"
                            className="block px-3 py-2 rounded-md text-base font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                        >
                            Try It Now
                        </Link>
                    </nav>
                </div>
            )}
            <ContactPopup isOpen={isContactOpen} onClose={() => setIsContactOpen(false)} />
        </header>
    )
}

