$(document).ready(function() {
    let gameNames = [];
    let searchTimeout = null;
    let recommendations = []; // To store for sorting
    let isSearching = false;

    // Load game names for autocomplete
    $.get('/game_names', function(data) {
        if (Array.isArray(data)) {
            gameNames = data;
            $('#game-count').text(data.length.toLocaleString());
        } else {
            showError('Error loading game names: ' + data.error);
        }
    }).fail(function() {
        showError('Failed to load game names.');
    });

    // Load popular games as grid
    loadPopularGames();

    // Autocomplete input
    $('#game-search').on('input', function() {
        const input = $(this).val().trim().toLowerCase();
        if (searchTimeout) clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => showSuggestions(input), 200);
    });

    function showSuggestions(input) {
        $('#suggestions').empty().hide();
        if (input.length < 2) return;

        // Fuzzy filter: prioritize startsWith, then includes
        const suggestions = gameNames
            .filter(name => name.toLowerCase().includes(input))
            .sort((a, b) => {
                const aLower = a.toLowerCase();
                const bLower = b.toLowerCase();
                if (aLower.startsWith(input) && !bLower.startsWith(input)) return -1;
                if (!aLower.startsWith(input) && bLower.startsWith(input)) return 1;
                return 0;
            })
            .slice(0, 15);

        if (suggestions.length > 0) {
            suggestions.forEach(suggestion => {
                const highlighted = suggestion.replace(new RegExp(input, 'gi'), match => `<span class="highlight">${match}</span>`);
                $('<div class="suggestion-item">')
                    .html(highlighted)
                    .on('click', function() {
                        $('#game-search').val(suggestion);
                        $('#suggestions').hide();
                        searchGame(suggestion);
                    })
                    .appendTo('#suggestions');
            });
            $('#suggestions').show();
        }
    }

    // Close suggestions on outside click
    $(document).on('click', function(e) {
        if (!$(e.target).closest('.search-bar').length) {
            $('#suggestions').hide();
        }
    });

    // Search button
    $('#search-btn').click(function() {
        const gameName = $('#game-search').val().trim();
        if (!gameName) {
            showError('Please enter a game name!');
            return;
        }
        searchGame(gameName);
    });

    // Enter key search
    $('#game-search').keypress(function(e) {
        if (e.which === 13) $('#search-btn').click();
    });

    function searchGame(gameName) {
        isSearching = true;
        $('#selected-section, #recommendations-section').addClass('d-none');
        $('#popular-section').fadeOut(500, function() {
            $(this).addClass('d-none');
        });
        showLoading(true);

        // Get game info
        $.ajax({
            url: '/get_game_info',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ game_name: gameName }),
            success: function(gameInfo) {
                if ($.isEmptyObject(gameInfo)) {
                    showError('Game not found! Try another name.');
                    resetToPopular();
                    return;
                }
                displaySelectedGame(gameInfo);
                $('#selected-section').removeClass('d-none').fadeIn(500);
            },
            error: function() {
                showError('Game not found! Please try another name.');
                resetToPopular();
            },
            complete: function() {
                showLoading(false);
            }
        });

        // Get recommendations
        $.ajax({
            url: '/recommend',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ game_name: gameName }),
            success: function(recs) {
                recommendations = recs;
                displayRecommendations(recs);
                $('#recommendations-section').removeClass('d-none').fadeIn(500);
                scrollToSection('#recommendations-section');
            },
            error: function() {
                showError('Error getting recommendations!');
                resetToPopular();
            }
        });
    }

    function resetToPopular() {
        isSearching = false;
        $('#popular-section').removeClass('d-none').fadeIn(500);
        $('#selected-section, #recommendations-section').fadeOut(500, function() {
            $(this).addClass('d-none');
        });
    }

    function displaySelectedGame(game) {
        const card = `
            <div class="game-card selected" data-link="${game['Link Game']}">
                <div class="img-container">
                    <img src="${game['Header image']}" alt="${game.Name}" class="game-img lazy" loading="lazy">
                </div>
                <div class="game-info">
                    <h3 class="game-title">${game.Name}</h3>
                    <p class="game-desc">${game['Short description']}</p>
                    <div class="game-meta">
                        <span class="game-genre"><i class="fas fa-tag"></i> ${game.Genres}</span>
                        <span class="game-rating"><i class="fas fa-star"></i> ${game.Rating} (${formatNumber(game['Total Reviews'])} reviews)</span>
                    </div>
                </div>
            </div>
        `;
        const $card = $(card).click(() => window.open(game['Link Game'], '_blank'));
        $('#selected-game').html($card).fadeIn(500);
    }

    function displayRecommendations(recs) {
        $('#game-list').empty();
        recs.forEach(game => {
            const hasVideo = game.Movies && game.Movies !== '';
            const card = `
                <div class="col-lg-4 col-md-6 mb-4">
                    <div class="game-card" data-link="${game['Link Game']}">
                        <div class="img-container">
                            <img src="${game['Header image']}" alt="${game.Name}" class="game-img lazy" loading="lazy">
                            ${hasVideo ? `
                                <div class="video-container">
                                    <video class="trailer-video" muted loop preload="none">
                                        <source src="${game.Movies.split(',')[0]}" type="video/mp4">
                                    </video>
                                </div>
                            ` : ''}
                        </div>
                        <div class="game-info">
                            <h3 class="game-title">${game.Name}</h3>
                            <p class="game-desc">${game['Short description']}</p>
                            <div class="game-meta">
                                <span class="game-genre"><i class="fas fa-gamepad"></i> ${game.Genres}</span>
                                <span class="game-rating"><i class="fas fa-star"></i> ${game.Rating}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            const $card = $(card);
            if (hasVideo) {
                $card.find('.img-container').hover(
                    function() {
                        const video = $(this).find('.trailer-video')[0];
                        video.play().catch(() => {});
                        $(this).find('.video-container').fadeIn(300);
                    },
                    function() {
                        const video = $(this).find('.trailer-video')[0];
                        video.pause();
                        video.currentTime = 0;
                        $(this).find('.video-container').fadeOut(300);
                    }
                );
            }
            $card.click(() => window.open(game['Link Game'], '_blank'));
            $('#game-list').append($card);
        });
        $('#game-list').fadeIn(500);
    }

    // Sort select change
    $('#sort-select').change(function() {
        const sortBy = $(this).val();
        let sortedRecs = [...recommendations];
        if (sortBy === 'rating') {
            sortedRecs.sort((a, b) => parseFloat(b.Rating) - parseFloat(a.Rating));
        } else if (sortBy === 'reviews') {
            sortedRecs.sort((a, b) => b['Total Reviews'] - a['Total Reviews']);
        }
        displayRecommendations(sortedRecs);
    });

    // Random button
    $('#random-btn').click(function() {
        $(this).prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Loading...');
        $.get('/random_game', function(game) {
            $('#game-search').val(game.Name);
            searchGame(game.Name);
        }).fail(function() {
            showError('Error getting random game!');
        }).always(function() {
            $('#random-btn').prop('disabled', false).html('<i class="fas fa-dice"></i> Random');
        });
    });

    function loadPopularGames() {
        $.get('/popular_games?num=9', function(games) {  // Lấy 9 games cho grid 3x3
            const popularList = $('#popular-games-list');
            games.forEach(game => {
                const card = `
                    <div class="col-lg-4 col-md-6 mb-4">
                        <div class="game-card" data-name="${game.Name}">
                            <div class="img-container">
                                <img src="${game['Header image']}" alt="${game.Name}" class="game-img lazy" loading="lazy">
                            </div>
                            <div class="game-info">
                                <h3 class="game-title">${game.Name}</h3>
                                <p class="game-desc">${game['Short description']}</p>
                                <div class="game-meta">
                                    <span class="game-genre"><i class="fas fa-tag"></i> ${game.Genres}</span>
                                    <span class="game-rating"><i class="fas fa-star"></i> ${game.Rating} (${formatNumber(game['Total Reviews'])})</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                const $card = $(card).click(function() {
                    const gameName = $(this).find('.game-card').data('name');
                    $('#game-search').val(gameName);
                    searchGame(gameName);
                });
                popularList.append($card);
            });
        }).fail(function() {
            showError('Error loading popular games!');
        });
    }

    // Utility functions
    function formatNumber(num) {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    }

    function showLoading(show) {
        if (show) {
            $('#loading').removeClass('d-none').css('opacity', 0).animate({opacity: 1}, 300);
        } else {
            $('#loading').animate({opacity: 0}, 300, function() {
                $(this).addClass('d-none');
            });
        }
    }

    function showError(message) {
        // Use Bootstrap modal for errors
        $('body').append(`
            <div class="modal fade" id="errorModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content bg-dark text-white">
                        <div class="modal-header border-0">
                            <h5 class="modal-title">Error</h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">${message}</div>
                        <div class="modal-footer border-0">
                            <button type="button" class="btn btn-neon" data-bs-dismiss="modal">OK</button>
                        </div>
                    </div>
                </div>
            </div>
        `);
        $('#errorModal').modal('show').on('hidden.bs.modal', function() { $(this).remove(); });
    }

    function scrollToSection(sectionId) {
        $('html, body').animate({
            scrollTop: $(sectionId).offset().top - 100
        }, 800);
    }

    // Back to top
    $(window).scroll(function() {
        $('#back-to-top').toggle($(this).scrollTop() > 300);
    });
    $('#back-to-top').click(function() {
        $('html, body').animate({ scrollTop: 0 }, 600);
    });

    // Smooth scroll for anchors
    $('a[href^="#"]').click(function(e) {
        e.preventDefault();
        const target = $(this.hash);
        if (target.length) {
            $('html, body').animate({ scrollTop: target.offset().top - 80 }, 600);
        }
    });
});