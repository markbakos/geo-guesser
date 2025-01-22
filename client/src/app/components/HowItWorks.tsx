"use client"

import { useState, useEffect } from "react"
import { Upload, Cpu, MapPin } from "lucide-react"

const steps = [
    {
        id: "01",
        name: "Upload",
        description: "Select and upload any image from your device.",
        icon: Upload,
    },
    {
        id: "02",
        name: "Process",
        description: "Our AI analyzes the image using advanced deep learning techniques.",
        icon: Cpu,
    },
    {
        id: "03",
        name: "Predict",
        description: "Receive a prediction of where the image was likely taken.",
        icon: MapPin,
    },
]

export default function HowItWorks() {
    const [activeStep, setActiveStep] = useState(0)

    useEffect(() => {
        const interval = setInterval(() => {
            setActiveStep((current) => (current + 1) % steps.length)
        }, 3000)

        return () => clearInterval(interval)
    }, [])

    return (
        <div id="how-it-works" className="bg-gray-50 py-24">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="lg:text-center">
                    <h2 className="text-base text-blue-600 font-semibold tracking-wide uppercase">How It Works</h2>
                    <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
                        Simple, Fast, and Accurate
                    </p>
                </div>

                <div className="mt-16">
                    <div className="lg:grid lg:grid-cols-2 lg:gap-8 lg:items-center">
                        <div className="relative">
                            <div className="relative pb-10 space-y-10">
                                {steps.map((step, index) => (
                                    <div
                                        key={step.id}
                                        className={`relative flex items-center space-x-4 z-10 ${index === activeStep ? "opacity-100" : "opacity-50"}`}
                                    >
                                        <div
                                            className={`flex h-12 w-12 items-center justify-center rounded-full ${index === activeStep ? "bg-blue-500" : "bg-blue-300"} transition-colors duration-200`}
                                        >
                                            <step.icon className={`h-6 w-6 text-white ${index === activeStep ? "animate-bounce" : ""}`} />
                                        </div>
                                        <div className="min-w-0 flex-1">
                                            <h3 className="text-lg font-medium text-gray-900">{step.name}</h3>
                                            <p className="text-base text-gray-500">{step.description}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div className="absolute left-6 top-4 h-full w-0.5 bg-gray-200" aria-hidden="true"></div>
                        </div>
                        <div className="mt-10 lg:mt-0 relative">
                            <div className="relative h-64 overflow-hidden rounded-lg bg-white shadow">
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="text-center">
                                        <p className="mt-1 text-4xl font-extrabold text-gray-900 sm:text-5xl sm:tracking-tight lg:text-6xl">
                                            {activeStep + 1}
                                        </p>
                                        <p className="max-w-xs mx-auto mt-5 text-xl text-gray-500">{steps[activeStep].name}</p>
                                    </div>
                                </div>
                                <div className="absolute bottom-0 left-0 right-0">
                                    <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200">
                                        <div
                                            style={{ width: `${((activeStep + 1) / steps.length) * 100}%` }}
                                            className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-blue-500 transition-all duration-500 ease-in-out"
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

