document.addEventListener("DOMContentLoaded", function () {

    /* =========================================================
       IMAGE PREVIEW
    ========================================================= */

    function previewFile(input, previewId) {
        const preview = document.getElementById(previewId);

        if (!preview || !input.files || !input.files[0]) {
            return;
        }

        const file = input.files[0];

        // Check whether the selected file is an image
        if (!file.type.startsWith("image/")) {
            preview.innerHTML = `
                <span class="text-danger">
                    Please select a valid image file.
                </span>
            `;
            return;
        }

        const reader = new FileReader();

        reader.addEventListener("load", function (event) {
            preview.innerHTML = `
                <img 
                    src="${event.target.result}" 
                    alt="Image Preview"
                >
            `;
        });

        reader.readAsDataURL(file);
    }


    /* =========================================================
       CONTENT IMAGE PREVIEW
    ========================================================= */

    const contentInput = document.querySelector(
        'input[name="content"]'
    );

    if (contentInput) {
        contentInput.addEventListener("change", function () {
            previewFile(this, "contentPreview");
        });
    }


    /* =========================================================
       STYLE IMAGE PREVIEW
    ========================================================= */

    const styleInput = document.querySelector(
        'input[name="style"]'
    );

    if (styleInput) {
        styleInput.addEventListener("change", function () {
            previewFile(this, "stylePreview");
        });
    }


    /* =========================================================
       NEURAL NETWORK CANVAS
    ========================================================= */

    const canvas = document.getElementById("neuralCanvas");

    if (canvas) {

        const ctx = canvas.getContext("2d");

        let width;
        let height;
        let particles = [];

        const mouse = {
            x: null,
            y: null,
            radius: 150
        };


        /* -----------------------------------------------------
           RESIZE CANVAS
        ----------------------------------------------------- */

        function resizeCanvas() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }


        /* -----------------------------------------------------
           PARTICLE CLASS
        ----------------------------------------------------- */

        class Particle {

            constructor() {

                this.x = Math.random() * width;
                this.y = Math.random() * height;

                this.vx = (Math.random() - 0.5) * 0.5;
                this.vy = (Math.random() - 0.5) * 0.5;

                this.size = Math.random() * 3 + 1.5;

                this.hue = Math.floor(Math.random() * 10) * 36;

                this.density =
                    Math.random() * 30 + 1;
            }


            update() {

                this.x += this.vx;
                this.y += this.vy;


                /* Mouse interaction */

                if (mouse.x !== null && mouse.y !== null) {

                    const dx = mouse.x - this.x;
                    const dy = mouse.y - this.y;

                    const distance =
                        Math.sqrt(dx * dx + dy * dy);

                    if (distance < mouse.radius && distance > 0) {

                        const forceDirectionX =
                            dx / distance;

                        const forceDirectionY =
                            dy / distance;

                        const force =
                            (mouse.radius - distance) /
                            mouse.radius;

                        const directionX =
                            forceDirectionX *
                            force *
                            this.density;

                        const directionY =
                            forceDirectionY *
                            force *
                            this.density;

                        this.x -= directionX;
                        this.y -= directionY;
                    }
                }


                /* Screen boundaries */

                if (this.x < 0 || this.x > width) {
                    this.vx *= -1;
                }

                if (this.y < 0 || this.y > height) {
                    this.vy *= -1;
                }


                /* Slowly change particle color */

                this.hue =
                    (this.hue + 0.2) % 360;
            }


            draw() {

                ctx.fillStyle =
                    `hsla(${this.hue}, 70%, 60%, 0.3)`;

                ctx.beginPath();

                ctx.arc(
                    this.x,
                    this.y,
                    this.size,
                    0,
                    Math.PI * 2
                );

                ctx.fill();
            }
        }


        /* -----------------------------------------------------
           INITIALIZE PARTICLES
        ----------------------------------------------------- */

        function initParticles() {

            particles = [];

            /*
             * Number of particles depends on
             * screen width.
             */

            const particleCount =
                Math.floor(window.innerWidth / 15);

            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
        }


        /* -----------------------------------------------------
           DRAW CONNECTIONS
        ----------------------------------------------------- */

        function drawConnections() {

            for (let i = 0; i < particles.length; i++) {

                for (
                    let j = i + 1;
                    j < particles.length;
                    j++
                ) {

                    const p1 = particles[i];
                    const p2 = particles[j];

                    const distance =
                        Math.hypot(
                            p1.x - p2.x,
                            p1.y - p2.y
                        );

                    if (distance < 100) {

                        ctx.strokeStyle =
                            `hsla(
                                ${p1.hue},
                                70%,
                                60%,
                                ${0.15 * (1 - distance / 100)}
                            )`;

                        ctx.lineWidth = 1;

                        ctx.beginPath();

                        ctx.moveTo(
                            p1.x,
                            p1.y
                        );

                        ctx.lineTo(
                            p2.x,
                            p2.y
                        );

                        ctx.stroke();
                    }
                }
            }
        }


        /* -----------------------------------------------------
           ANIMATION LOOP
        ----------------------------------------------------- */

        function animate() {

            ctx.clearRect(
                0,
                0,
                width,
                height
            );

            particles.forEach(function (particle) {

                particle.update();
                particle.draw();

            });

            drawConnections();

            requestAnimationFrame(animate);
        }


        /* -----------------------------------------------------
           MOUSE EVENTS
        ----------------------------------------------------- */

        window.addEventListener(
            "mousemove",
            function (event) {

                mouse.x = event.clientX;
                mouse.y = event.clientY;

            }
        );


        window.addEventListener(
            "mouseout",
            function () {

                mouse.x = null;
                mouse.y = null;

            }
        );


        /* -----------------------------------------------------
           WINDOW RESIZE
        ----------------------------------------------------- */

        window.addEventListener(
            "resize",
            function () {

                resizeCanvas();
                initParticles();

            }
        );


        /* Start animation */

        resizeCanvas();
        initParticles();
        animate();
    }


    /* =========================================================
       ALPHA / STYLE STRENGTH SLIDER
    ========================================================= */

    const range =
        document.getElementById("alphaRange");

    const rangeValue =
        document.getElementById("rangeValue");


    function updateRange() {

        if (!range || !rangeValue) {
            return;
        }

        const value =
            parseFloat(range.value);

        const min =
            parseFloat(range.min) || 0;

        const max =
            parseFloat(range.max) || 1;

        const percentage =
            ((value - min) * 100) /
            (max - min);


        /* Update displayed value */

        rangeValue.textContent =
            value.toFixed(1);


        /* Move value bubble */

        const thumbWidth = 24;

        const offset =
            thumbWidth / 2 -
            (percentage * thumbWidth / 100);

        rangeValue.style.left =
            `calc(${percentage}% + ${offset}px)`;


        /* Update slider background */

        range.style.background =
            `linear-gradient(
                to right,
                #818cf8 0%,
                #c084fc ${percentage}%,
                rgba(30, 41, 59, 0.8) ${percentage}%,
                rgba(30, 41, 59, 0.8) 100%
            )`;
    }


    if (range) {

        range.addEventListener(
            "input",
            updateRange
        );

        updateRange();
    }


    /* =========================================================
       SHOW RANGE VALUE ON HOVER
    ========================================================= */

    const rangeWrap =
        document.querySelector(".range-wrap");

    if (rangeWrap && rangeValue) {

        rangeWrap.addEventListener(
            "mouseenter",
            function () {
                rangeValue.style.opacity = "1";
            }
        );

        rangeWrap.addEventListener(
            "mouseleave",
            function () {
                rangeValue.style.opacity = "0";
            }
        );
    }


    /* =========================================================
       LOADING OVERLAY
    ========================================================= */

    const uploadForm =
        document.getElementById("uploadForm");

    const loader =
        document.getElementById("loader");


    if (uploadForm && loader) {

        uploadForm.addEventListener(
            "submit",
            function (event) {

                /*
                 * Only show loader when both
                 * images have been selected.
                 */

                const content =
                    document.querySelector(
                        'input[name="content"]'
                    );

                const style =
                    document.querySelector(
                        'input[name="style"]'
                    );


                if (
                    content &&
                    style &&
                    content.files.length > 0 &&
                    style.files.length > 0
                ) {

                    loader.classList.add("active");
                }
            }
        );
    }


    /* =========================================================
       RESULT SECTION AUTO SCROLL
    ========================================================= */

    const resultSection =
        document.getElementById("resultSection");

    if (resultSection) {

        setTimeout(function () {

            resultSection.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });

        }, 300);
    }


    /* =========================================================
       NAVBAR SCROLL EFFECT
    ========================================================= */

    const navbar =
        document.querySelector(".navbar");

    if (navbar) {

        window.addEventListener(
            "scroll",
            function () {

                if (window.scrollY > 50) {

                    navbar.classList.add(
                        "navbar-scrolled"
                    );

                } else {

                    navbar.classList.remove(
                        "navbar-scrolled"
                    );
                }
            }
        );
    }


    /* =========================================================
       FADE-IN ANIMATION
    ========================================================= */

    const fadeElements =
        document.querySelectorAll(".fade-in-up");

    if ("IntersectionObserver" in window) {

        const observer =
            new IntersectionObserver(
                function (entries) {

                    entries.forEach(function (entry) {

                        if (entry.isIntersecting) {

                            entry.target.style.opacity = "1";
                            entry.target.style.transform =
                                "translateY(0)";

                            observer.unobserve(
                                entry.target
                            );
                        }

                    });

                },
                {
                    threshold: 0.1
                }
            );


        fadeElements.forEach(function (element) {
            observer.observe(element);
        });

    } else {

        /*
         * Fallback for older browsers
         */

        fadeElements.forEach(function (element) {

            element.style.opacity = "1";
            element.style.transform =
                "translateY(0)";

        });
    }

});


