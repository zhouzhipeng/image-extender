/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/index.html',
        destination: '/',
      },
    ]
  },
}

module.exports = nextConfig

