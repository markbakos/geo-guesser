"use client"

import { useEffect, useRef } from "react";
import Globe, { GlobeMethods } from "react-globe.gl";

export default function AnimatedGlobe() {
    const globeRef = useRef<GlobeMethods | undefined>(undefined)

    useEffect(() => {
        if (globeRef.current) {
            const globe = globeRef.current

            globe.controls().autoRotate = true
            globe.controls().autoRotateSpeed = 0.5
        }
    }, [])

    return (
        <Globe
            ref={globeRef}
            width={600}
            height={400}
            backgroundColor="rgba(0,0,0,0)"
            globeImageUrl="https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
        />
    )
}