export function getAboutPageContent(): string {
    return `
        <h1>About</h1>
        <p>This is a demonstration of a single-page application built with TypeScript and WebGPU.</p>
        <p>
            <a href="#" class="download-button">[Download PDF Resume]</a>
        </p>
        <ul>
            <li><a href="mailto:example@example.com">Email</a></li>
            <li><a href="https://github.com" target="_blank">GitHub</a></li>
            <li><a href="https://linkedin.com" target="_blank">LinkedIn</a></li>
        </ul>
    `;
}
