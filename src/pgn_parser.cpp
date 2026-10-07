#include "pgn_parser.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

//...............Constructors...............
PgnParser::PgnParser()
{
}

//...............Méthodes...............
bool PgnParser::parseFiles(const std::string& filename)
{
    std::cout << "Lecture du fichier pgn : " << filename << std::endl;
    // 1. Vérifier que l'extension est bien "pgn"
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos || filename.substr(dot_pos + 1) != "pgn")
    {
        std::cerr << "Erreur : Le fichier doit avoir l'extension .pgn" << std::endl;
        return false;
    }

    // 2. Vérifier que le fichier existe et l'ouvrir
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Nettoyage des éventuels retours chariots Windows (\r)
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        if (line.empty())
        {
            continue;
        }

        // 3. Séparer les tags et les coups
        if (line[0] == '[')
        {

            if (!this->raw_moves.empty())
            {
                std::cerr << "Erreur : Le fichier PGN contient plusieurs parties. "
                    << "Veuillez fournir un fichier avec une seule partie." << std::endl;
                file.close();
                return false;
            }

            // Format attendu : [Cle "Valeur"]
            size_t space_pos = line.find(' ');
            size_t first_quote = line.find('"');
            size_t second_quote = line.find('"', first_quote + 1);

            // S'assurer que la syntaxe est respectée avant d'extraire
            if (space_pos != std::string::npos && first_quote != std::string::npos && second_quote != std::string::npos)
            {
                std::string key = line.substr(1, space_pos - 1);
                std::string value = line.substr(first_quote + 1, second_quote - first_quote - 1);
                this->tags[key] = value;
            }
        }
        else
        {
            // Si ça ne commence pas par '[', c'est la suite des coups bruts.
            // On ajoute un espace pour ne pas coller la fin d'une ligne au début de la suivante.
            if (!this->raw_moves.empty())
            {
                this->raw_moves += " ";
            }
            this->raw_moves += line;
        }
    }

    file.close();
    //std::cout << "Lecture reussie. \n";
    return true;
}

//...............Getters...............
const std::unordered_map<std::string, std::string>& PgnParser::getTags() const
{
    return this->tags;
}

std::string PgnParser::getRawMoves() const
{
    return this->raw_moves;
}


std::vector<std::string> PgnParser::extractMoves() const
{
    std::vector<std::string> moves;
    std::stringstream ss(this->raw_moves);
    std::string token;

    // L'opérateur >> extrait les mots séparés par des espaces ou sauts de ligne
    while (ss >> token)
    {
        // 1. Ignorer les marqueurs de fin de partie
        if (token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*")
        {
            continue;
        }

        // 2. Traitement des jetons contenant un point ("1.", "1.Nf3", "12...Nxd5")
        size_t dot_pos = token.find_last_of('.');
        if (dot_pos != std::string::npos)
        {
            std::string actual_move = token.substr(dot_pos + 1);
            if (!actual_move.empty())
            {
                moves.push_back(actual_move);
            }
            continue;
        }

        // 3. Ajouter le coup valide (sans point) à la liste
        moves.push_back(token);
    }

    return moves;
}