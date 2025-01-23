import {Globe, Lock, GraduationCap} from "lucide-react"

const features = [
    {
        name: "Global Coverage",
        description: "Our AI model has been trained on ~50K of images from around the world.",
        icon: Globe,
    },
    {
        name: "Study Project & Open Source",
        description: "This project's source code is available on GitHub and was made by a student.",
        icon: GraduationCap,
    },
    {
        name: "Privacy Focused",
        description: "Your uploads are processed securely and never stored on our servers.",
        icon: Lock,
    },
]

export default function Features() {
    return (
        <div id="features" className="py-24 bg-white">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="lg:text-center">
                    <h2 className="text-base text-blue-600 font-semibold tracking-wide uppercase">Features</h2>
                    <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
                        Why Use LocationGuesser?
                    </p>
                </div>

                <div className="mt-20">
                    <dl className="space-y-10 md:space-y-0 md:grid md:grid-cols-3 md:gap-x-8 md:gap-y-10">
                        {features.map((feature) => (
                            <div key={feature.name} className="relative">
                                <dt>
                                    <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-blue-500 text-white">
                                        <feature.icon className="h-6 w-6" aria-hidden="true" />
                                    </div>
                                    <p className="ml-16 text-lg leading-6 font-medium text-gray-900">{feature.name}</p>
                                </dt>
                                <dd className="mt-2 ml-16 text-base text-gray-500">{feature.description}</dd>
                            </div>
                        ))}
                    </dl>
                </div>
            </div>
        </div>
    )
}

