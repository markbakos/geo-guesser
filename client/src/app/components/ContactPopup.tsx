"use client"

import { useEffect, useRef } from "react"
import { X, Linkedin, Github, Mail, Globe } from "lucide-react"

interface ContactPopupProps {
    isOpen: boolean
    onClose: () => void
}

export default function ContactPopup({ isOpen, onClose }: ContactPopupProps) {
    const popupRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (popupRef.current && !popupRef.current.contains(event.target as Node)) {
                onClose()
            }
        }

        if (isOpen) {
            document.addEventListener("mousedown", handleClickOutside)
        }

        return () => {
            document.removeEventListener("mousedown", handleClickOutside)
        }
    }, [isOpen, onClose])

    if (!isOpen) return null

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[99999] p-4">
            <div ref={popupRef} className="bg-white rounded-lg shadow-xl p-6 w-full max-w-sm">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-bold text-gray-900">Contact Me</h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
                        <X size={24} />
                    </button>
                </div>
                <div className="space-y-4">
                    <a
                        href="https://github.com/markbakos"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center text-gray-700 hover:text-gray-900"
                    >
                        <Github className="mr-2" size={20}/>
                        GitHub
                    </a>
                    <a
                        href="https://markbakos.onrender.com"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center text-purple-600 hover:text-purple-800"
                    >
                        <Globe className="mr-2" size={20}/>
                        Portfolio
                    </a>
                    <a
                        href="https://www.linkedin.com/in/markbakos"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center text-blue-600 hover:text-blue-800"
                    >
                        <Linkedin className="mr-2" size={20}/>
                        LinkedIn
                    </a>
                    <a href="mailto:markbakosss@gmail.com"
                       className="flex items-center text-green-600 hover:text-green-800">
                        <Mail className="mr-2" size={20}/>
                        Email
                    </a>
                </div>
            </div>
        </div>
    )
}

