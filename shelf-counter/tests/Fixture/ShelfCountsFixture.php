<?php
declare(strict_types=1);

namespace App\Test\Fixture;

use Cake\TestSuite\Fixture\TestFixture;

/**
 * ShelfCountsFixture
 */
class ShelfCountsFixture extends TestFixture
{
    /**
     * Init method
     *
     * @return void
     */
    public function init(): void
    {
        $this->records = [
            [
                'id' => 1,
                'key' => 'Lorem ipsum dolor sit amet',
                'count' => 1,
                'created' => '2025-10-06 16:31:03',
                'modified' => '2025-10-06 16:31:03',
            ],
        ];
        parent::init();
    }
}
