/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: {
				page: 'var(--color-bg-page)',
				brand: 'var(--color-text-brand)',
				default: 'var(--color-text-default)',
				weak: 'var(--color-text-weak)',
			},
      fontFamily: {
        heading: 'var(--font-heading)',
        body: 'var(--font-body)',
      }
		},
	},
	plugins: [
    require('@tailwindcss/typography'),
  ],
}