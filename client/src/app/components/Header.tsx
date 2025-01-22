import Link from "next/link"
import { MapPin } from "lucide-react"

export default function Header() {
    return (
        <header className="bg-white shadow-sm">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center">
                        <MapPin className="h-8 w-8 text-blue-600" />
                        <span className="ml-2 text-xl font-bold text-gray-900 ">LocationGuesser</span>
                    </div>
                    <nav>
                        <ul className="flex space-x-4">
                            <li>
                                <Link href="/" className="text-gray-600 hover:text-gray-900">
                                    Home
                                </Link>
                            </li>
                            <li>
                                <Link href="#features" className="text-gray-600 hover:text-gray-900">
                                    Features
                                </Link>
                            </li>
                            <li>
                                <Link href="#how-it-works" className="text-gray-600 hover:text-gray-900">
                                    How It Works
                                </Link>
                            </li>
                            <li>
                                <Link href="#" className="text-blue-600 hover:text-blue-700 font-medium">
                                    Try It Now
                                </Link>
                            </li>
                        </ul>
                    </nav>
                </div>
            </div>
        </header>
    )
}

