$(document).ready(function() {
    let gameNames = [];
    let searchTimeout = null;
    let recommendations = [];
    let seedGames = [];

    $.get('/game_names', function(data) {
        if (Array.isArray(data)) {
            gameNames = data;
            $('#game-count').text(data.length.toLocaleString());
        } else {
            showError('Error loading game names: ' + (data && data.error ? data.error : 'unknown'));
        }
    }).fail(function() {
        showError('Failed to load game names.');
    });

    $.get('/genres', function(data) {
        if (Array.isArray(data)) {
            const $sel = $('#filter-genre');
            data.forEach(g => {
                $sel.append($('<option>').val(g).text(g));
            });
        }
    });

    loadPopularGames();

    $('#game-search').on('input', function() {
        const input = $(this).val().trim().toLowerCase();
        if (searchTimeout) clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => showSuggestions(input, '#suggestions', (name) => {
            $('#game-search').val(name);
            $('#suggestions').hide();
            searchGame(name);
        }), 200);
    });

    function showSuggestions(input, containerSel, onPick) {
        const $box = $(containerSel);
        $box.empty().hide();
        if (input.length < 2) return;

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
                const highlighted = suggestion.replace(
                    new RegExp(input.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'),
                    match => `<span class="highlight">${match}</span>`
                );
                $('<div class="suggestion-item">')
                    .html(highlighted)
                    .on('click', function() { onPick(suggestion); })
                    .appendTo($box);
            });
            $box.show();
        }
    }

    $(document).on('click', function(e) {
        if (!$(e.target).closest('.search-bar, .seed-row').length) {
            $('#suggestions').hide();
        }
    });

    $('#search-btn').click(function() {
        const gameName = $('#game-search').val().trim();
        if (!gameName) {
            showError('Please enter a game name!');
            return;
        }
        searchGame(gameName);
    });

    $('#game-search').keypress(function(e) {
        if (e.which === 13) $('#search-btn').click();
    });

    $('#add-seed-btn').click(function() {
        const name = ($('#seed-input').val() || $('#game-search').val() || '').trim();
        if (!name) return;
        if (!seedGames.some(s => s.toLowerCase() === name.toLowerCase())) {
            seedGames.push(name);
            renderSeedChips();
        }
        $('#seed-input').val('');
    });

    $('#seed-input').keypress(function(e) {
        if (e.which === 13) {
            e.preventDefault();
            $('#add-seed-btn').click();
        }
    });

    function renderSeedChips() {
        const $chips = $('#seed-chips').empty();
        seedGames.forEach((name, idx) => {
            const $chip = $(`<span class="seed-chip">${name} <i class="fas fa-times"></i></span>`);
            $chip.find('i').on('click', function(e) {
                e.stopPropagation();
                seedGames.splice(idx, 1);
                renderSeedChips();
            });
            $chips.append($chip);
        });
    }

    function collectFilters() {
        const minRatingPct = parseFloat($('#filter-min-rating').val());
        const minReviews = parseInt($('#filter-min-reviews').val(), 10);
        const maxPriceRaw = $('#filter-max-price').val();
        const genre = $('#filter-genre').val();
        const payload = {
            min_rating: isNaN(minRatingPct) ? 0.65 : minRatingPct / 100,
            min_reviews: isNaN(minReviews) ? 5000 : minReviews,
            multiplayer_only: $('#filter-multiplayer').is(':checked'),
            num_recommendations: 9,
        };
        if (maxPriceRaw !== '' && maxPriceRaw != null) {
            const mp = parseFloat(maxPriceRaw);
            if (!isNaN(mp)) payload.max_price = mp;
        }
        if (genre) payload.genres = [genre];
        return payload;
    }

    function searchGame(gameName) {
        // Primary game + multi-seeds
        const seeds = [gameName];
        seedGames.forEach(s => {
            if (s.toLowerCase() !== gameName.toLowerCase()) seeds.push(s);
        });

        $('#selected-section, #recommendations-section').addClass('d-none');
        $('#popular-section').fadeOut(300, function() {
            $(this).addClass('d-none');
        });
        showLoading(true);

        // Selected game info (primary)
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
                displaySelectedGame(gameInfo, seeds);
                $('#selected-section').removeClass('d-none').hide().fadeIn(400);
            },
            error: function() {
                showError('Game not found! Please try another name.');
                resetToPopular();
            }
        });

        const payload = Object.assign({ game_names: seeds }, collectFilters());

        $.ajax({
            url: '/recommend',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(payload),
            success: function(recs) {
                recommendations = Array.isArray(recs) ? recs : [];
                if (!recommendations.length) {
                    showError('No recommendations matched your filters. Try relaxing min reviews/rating.');
                }
                displayRecommendations(recommendations);
                $('#recommendations-section').removeClass('d-none').hide().fadeIn(400);
                scrollToSection('#recommendations-section');
            },
            error: function() {
                showError('Error getting recommendations!');
                resetToPopular();
            },
            complete: function() {
                showLoading(false);
            }
        });
    }

    function resetToPopular() {
        $('#popular-section').removeClass('d-none').fadeIn(400);
        $('#selected-section, #recommendations-section').fadeOut(300, function() {
            $(this).addClass('d-none');
        });
        showLoading(false);
    }

    $('#back-home-btn').click(function() {
        resetToPopular();
        scrollToSection('#popular-section');
    });

    function displaySelectedGame(game, seeds) {
        let seedsHtml = '';
        if (seeds && seeds.length > 1) {
            seedsHtml = `<p class="seed-note"><i class="fas fa-layer-group"></i> Multi-seed: ${seeds.map(s => escapeHtml(s)).join(' · ')}</p>`;
        }
        const card = `
            <div class="game-card selected" data-link="${game['Link Game']}">
                <div class="img-container">
                    <img src="${game['Header image']}" alt="${escapeHtml(game.Name)}" class="game-img lazy" loading="lazy">
                </div>
                <div class="game-info">
                    <h3 class="game-title">${escapeHtml(game.Name)}</h3>
                    <p class="game-desc">${escapeHtml(game['Short description'] || '')}</p>
                    <div class="game-meta">
                        <span class="game-genre"><i class="fas fa-tag"></i> ${escapeHtml(String(game.Genres || ''))}</span>
                        <span class="game-rating"><i class="fas fa-star"></i> ${escapeHtml(game.Rating)} (${formatNumber(game['Total Reviews'])} reviews)</span>
                    </div>
                    ${seedsHtml}
                </div>
            </div>
        `;
        const $card = $(card).click(() => window.open(game['Link Game'], '_blank'));
        $('#selected-game').html($card);
    }

    function displayRecommendations(recs) {
        $('#game-list').empty();
        if (!recs || !recs.length) {
            $('#game-list').html('<p class="text-center text-muted">No recommendations found.</p>');
            return;
        }
        recs.forEach(game => {
            const hasVideo = game.Movies && game.Movies !== '';
            const exp = game.explanation || {};
            const why = exp.summary || '';
            const sharedTags = (exp.shared_tags || []).slice(0, 5);
            const tagsHtml = sharedTags.length
                ? `<div class="tag-chips">${sharedTags.map(t => `<span class="tag-chip">${escapeHtml(t)}</span>`).join('')}</div>`
                : '';
            const simPct = ((game.Similarity || 0) * 100).toFixed(1);
            const qScore = (game['Quality Score'] != null) ? game['Quality Score'].toFixed(3) : '—';

            const card = `
                <div class="col-lg-4 col-md-6 mb-4">
                    <div class="game-card" data-link="${game['Link Game']}">
                        <div class="img-container">
                            <img src="${game['Header image']}" alt="${escapeHtml(game.Name)}" class="game-img lazy" loading="lazy">
                            ${hasVideo ? `
                                <div class="video-container">
                                    <video class="trailer-video" muted loop playsinline preload="none">
                                        <source src="${String(game.Movies).split(',')[0]}" type="video/mp4">
                                    </video>
                                </div>
                            ` : ''}
                            <div class="score-badge">${simPct}% match</div>
                        </div>
                        <div class="game-info">
                            <h3 class="game-title">${escapeHtml(game.Name)}</h3>
                            <p class="game-desc">${escapeHtml(game['Short description'] || '')}</p>
                            ${why ? `<p class="why-text"><i class="fas fa-lightbulb"></i> ${escapeHtml(why)}</p>` : ''}
                            ${tagsHtml}
                            <div class="game-meta">
                                <span class="game-genre"><i class="fas fa-gamepad"></i> ${escapeHtml(String(game.Genres || ''))}</span>
                                <span class="game-rating"><i class="fas fa-star"></i> ${escapeHtml(game.Rating)} · Q ${qScore}</span>
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
                        if (video) {
                            video.play().catch(() => {});
                            $(this).find('.video-container').css('opacity', 1);
                        }
                    },
                    function() {
                        const video = $(this).find('.trailer-video')[0];
                        if (video) {
                            video.pause();
                            video.currentTime = 0;
                            $(this).find('.video-container').css('opacity', 0);
                        }
                    }
                );
            }
            $card.find('.game-card').click(function(e) {
                if (game['Link Game']) {
                    window.open(game['Link Game'], '_blank', 'noopener,noreferrer');
                }
            });
            $('#game-list').append($card);
        });
    }

    $('#sort-select').change(function() {
        const sortBy = $(this).val();
        let sortedRecs = [...recommendations];
        if (sortBy === 'rating') {
            sortedRecs.sort((a, b) => parseFloat(b.Rating) - parseFloat(a.Rating));
        } else if (sortBy === 'reviews') {
            sortedRecs.sort((a, b) => b['Total Reviews'] - a['Total Reviews']);
        } else if (sortBy === 'similarity') {
            sortedRecs.sort((a, b) => (b.Similarity || 0) - (a.Similarity || 0));
        } else if (sortBy === 'quality') {
            sortedRecs.sort((a, b) => (b['Quality Score'] || 0) - (a['Quality Score'] || 0));
        }
        displayRecommendations(sortedRecs);
    });

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
        $.get('/popular_games?num=9', function(games) {
            const popularList = $('#popular-games-list').empty();
            games.forEach(game => {
                const card = `
                    <div class="col-lg-4 col-md-6 mb-4">
                        <div class="game-card" data-name="${escapeHtml(game.Name)}">
                            <div class="img-container">
                                <img src="${game['Header image']}" alt="${escapeHtml(game.Name)}" class="game-img lazy" loading="lazy">
                            </div>
                            <div class="game-info">
                                <h3 class="game-title">${escapeHtml(game.Name)}</h3>
                                <p class="game-desc">${escapeHtml(game['Short description'] || '')}</p>
                                <div class="game-meta">
                                    <span class="game-genre"><i class="fas fa-tag"></i> ${escapeHtml(String(game.Genres || ''))}</span>
                                    <span class="game-rating"><i class="fas fa-star"></i> ${escapeHtml(game.Rating)} (${formatNumber(game['Total Reviews'])})</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                const $wrap = $(card);
                $wrap.find('.game-card').click(function() {
                    const gameName = $(this).data('name');
                    $('#game-search').val(gameName);
                    searchGame(gameName);
                });
                popularList.append($wrap);
            });
        }).fail(function() {
            showError('Error loading popular games!');
        });
    }

    function formatNumber(num) {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return String(num);
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function showLoading(show) {
        if (show) {
            $('#loading').removeClass('d-none').css('opacity', 0).animate({ opacity: 1 }, 200);
        } else {
            $('#loading').animate({ opacity: 0 }, 200, function() {
                $(this).addClass('d-none');
            });
        }
    }

    function showError(message) {
        $('body').append(`
            <div class="modal fade" id="errorModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content bg-dark text-white">
                        <div class="modal-header border-0">
                            <h5 class="modal-title">Notice</h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">${escapeHtml(message)}</div>
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
        const $el = $(sectionId);
        if ($el.length) {
            $('html, body').animate({ scrollTop: $el.offset().top - 100 }, 600);
        }
    }

    $(window).scroll(function() {
        $('#back-to-top').toggle($(this).scrollTop() > 300);
    });
    $('#back-to-top').click(function() {
        $('html, body').animate({ scrollTop: 0 }, 600);
    });
});
