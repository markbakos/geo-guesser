"use client"

import {useEffect, useRef, useState} from "react";
import Globe, { GlobeMethods } from "react-globe.gl";

export default function AnimatedGlobe() {
    const globeRef = useRef<GlobeMethods | undefined>(undefined)

    const [dimensions, setDimensions] = useState({ width: 300, height: 300 })

    useEffect(() => {
        if (globeRef.current) {
            const globe = globeRef.current

            globe.controls().autoRotate = true
            globe.controls().autoRotateSpeed = 0.5
        }

        const updateDimensions = () => {
            const width = Math.min(500, window.innerWidth - 40)
            setDimensions({ width, height: width })
        }

        updateDimensions()
        window.addEventListener("resize", updateDimensions)

        return () => window.removeEventListener("resize", updateDimensions)
    }, [])

    return (
        <Globe
            ref={globeRef}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="rgba(0,0,0,0)"
            globeImageUrl="https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
        />
    )
}