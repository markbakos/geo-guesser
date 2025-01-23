import {MapPin} from "lucide-react"
import Header from "./components/Header"
import AnimatedGlobe from "./components/Globe"
import HowItWorks from "@/app/components/HowItWorks";
import Features from "@/app/components/Features";


export default function Home() {
    return (
        <div className="min-h-screen flex flex-col scroll-smooth">
            <Header />
            <main>
                <div className="bg-gradient-to-r from-blue-50 to-indigo-100">
                    <div
                        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 flex flex-col lg:flex-row items-center">
                        <div className="flex-1 text-center lg:text-left mb-10 lg:mb-0">
                            <h1 className="text-4xl font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                                Discover Where Your Images Were Taken
                            </h1>
                            <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto lg:mx-0 md:mt-5 md:text-xl">
                                Upload any image and our AI will predict its location. Fast, accurate, and completely
                                free.
                            </p>
                            <div className="mt-8 sm:mt-10">
                                <a
                                    href="/predict"
                                    className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                                >
                                    Get Started
                                </a>
                            </div>
                        </div>
                        <div className="flex-1 flex justify-center lg:justify-end">
                            <div className="w-full max-w-[600px] h-[400px]">
                                <AnimatedGlobe/>
                            </div>
                        </div>
                    </div>
                </div>
                <Features />
                <HowItWorks />
            </main>
            <footer className="bg-white">
                <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center">
                        <div className="flex items-center">
                            <MapPin className="h-8 w-8 text-blue-600"/>
                            <span className="ml-2 text-xl font-bold text-gray-900">LocationGuesser</span>
                        </div>
                        <nav className="-mx-5 -my-2 flex flex-wrap justify-center">
                            <div className="px-5 py-2">
                                <a href="https://github.com/markbakos" target="_blank" className="text-base text-gray-500 hover:text-gray-900">
                                    GitHub
                                </a>
                            </div>
                            <div className="px-5 py-2">
                                <a href="https://www.linkedin.com/in/markbakos/" target="_blank" className="text-base text-gray-500 hover:text-gray-900">
                                    LinkedIn
                                </a>
                            </div>
                            <div className="px-5 py-2">
                                <a href="mailto:markbakosss@gmail.com" target="_blank" className="text-base text-gray-500 hover:text-gray-900">
                                    Mail
                                </a>
                            </div>
                            <div className="px-5 py-2">
                                <a href="https://markbakos.onrender.com/" target="_blank" className="text-base text-gray-500 hover:text-gray-900">
                                    Portfolio
                                </a>
                            </div>
                        </nav>
                    </div>
                </div>
            </footer>
        </div>
    )
}