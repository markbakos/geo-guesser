"use client"

import { useEffect, useRef, useState } from "react"
import "leaflet/dist/leaflet.css"
import type * as Leaflet from "leaflet"

interface PredictionMapProps {
    prediction: [number, number] | null
}

export default function PredictionMap({ prediction }: PredictionMapProps) {
    const mapRef = useRef<Leaflet.Map | null>(null)
    const markerRef = useRef<Leaflet.Marker | null>(null)
    const mapContainerRef = useRef<HTMLDivElement | null>(null)
    const [L, setL] = useState<typeof Leaflet | null>(null)

    useEffect(() => {
        import("leaflet").then((leaflet) => {
            setL(leaflet)

            leaflet.Icon.Default.mergeOptions({
                iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
                iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
                shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
            })
        })
    }, [])

    useEffect(() => {
        if (!L || !mapContainerRef.current || mapRef.current) return

        mapRef.current = L.map(mapContainerRef.current).setView([0, 0], 2)
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        }).addTo(mapRef.current)
    }, [L])

    useEffect(() => {
        if (!L || !prediction || !mapRef.current) return

        const [lat, lng] = prediction
        const newLatLng = new L.LatLng(lat, lng)

        if (markerRef.current) {
            markerRef.current.setLatLng(newLatLng)
        } else {
            markerRef.current = L.marker(newLatLng).addTo(mapRef.current)
        }

        mapRef.current.setView(newLatLng, 8)
    }, [L, prediction])

    return <div ref={mapContainerRef} className="h-full min-h-[400px] rounded-lg shadow-md" />
}