// Navbar scroll effect
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Intersection Observer for scroll animations
const fadeElements = document.querySelectorAll('.reveal');

const appearOptions = {
    threshold: 0.15,
    rootMargin: "0px 0px -50px 0px"
};

const appearOnScroll = new IntersectionObserver(function (entries, observer) {
    entries.forEach(entry => {
        if (!entry.isIntersecting) {
            return;
        } else {
            entry.target.classList.add('active');
            observer.unobserve(entry.target);
        }
    });
}, appearOptions);

fadeElements.forEach(element => {
    appearOnScroll.observe(element);
});

// Hero Canvas Video Sequence
const canvas = document.getElementById('hero-canvas');
if (canvas) {
    const context = canvas.getContext('2d');

    // Set canvas dimensions (matches the original video aspect ratio roughly)
    canvas.width = 1920;
    canvas.height = 1080;

    const frameCount = 40;
    const images = [];
    const imagePath = 'assets/ezgif-framesF1-jpg/ezgif-frame-';

    // Preload images
    let loadedImages = 0;
    for (let i = 1; i <= frameCount; i++) {
        const img = new Image();
        // Pad single/double digits to match 001, 002 format
        const frameNum = i.toString().padStart(3, '0');
        img.src = `${imagePath}${frameNum}.jpg`;
        img.onload = () => {
            loadedImages++;
            // Draw first frame immediately if it's loaded
            if (i === 1) {
                context.drawImage(img, 0, 0, canvas.width, canvas.height);
            }
        };
        images.push(img);
    }

    // Animation Loop
    let currentFrame = 0;
    let lastDrawTime = 0;
    const fpsInterval = 1000 / 24; // ~24 FPS playback

    function drawSequence(timestamp) {
        if (!lastDrawTime) lastDrawTime = timestamp;

        const elapsed = timestamp - lastDrawTime;

        // Start animation only when most images are loaded
        if (loadedImages > frameCount / 2 && elapsed > fpsInterval) {
            lastDrawTime = timestamp - (elapsed % fpsInterval);

            // Draw current frame
            if (images[currentFrame] && images[currentFrame].complete) {
                context.drawImage(images[currentFrame], 0, 0, canvas.width, canvas.height);
            }

            // Loop back to start
            currentFrame = (currentFrame + 1) % frameCount;
        }

        requestAnimationFrame(drawSequence);
    }

    requestAnimationFrame(drawSequence);
}

// Fast number counter for stats (Optional Enhancement)
const statNumbers = document.querySelectorAll('.stat-number');

// Mobile menu toggle
const hamburger = document.querySelector('.hamburger');
const navLinks = document.querySelector('.nav-links');

if (hamburger) {
    hamburger.addEventListener('click', () => {
        // Toggle mobile menu logic here if needed
        alert("Mobile menu clicked (implement expansion logic)");
    });
}

// Theme Toggle Logic
const themeToggleBtn = document.getElementById('theme-toggle');
const themeIcon = themeToggleBtn ? themeToggleBtn.querySelector('i') : null;

// Check saved theme or system preference
const savedTheme = localStorage.getItem('theme');
const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;

if (savedTheme === 'light' || (!savedTheme && prefersLight)) {
    document.documentElement.setAttribute('data-theme', 'light');
    if (themeIcon) {
        themeIcon.classList.remove('fa-moon');
        themeIcon.classList.add('fa-sun');
    }
}

if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        let newTheme = 'dark';

        if (!currentTheme || currentTheme === 'dark') {
            newTheme = 'light';
            themeIcon.classList.remove('fa-moon');
            themeIcon.classList.add('fa-sun');
        } else {
            newTheme = 'dark';
            themeIcon.classList.remove('fa-sun');
            themeIcon.classList.add('fa-moon');
        }

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });
}
